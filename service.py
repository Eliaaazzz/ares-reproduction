"""Black-box service model wrapper.

The service model is CLIP ViT-B/16 used as a zero-shot classifier over the
target classes (one CoOp-style template per dataset). To keep the closed-box
contract honest, the only thing the rest of the code is allowed to call is
predict_proba(), which returns class probabilities and counts queries --
exactly what a commercial classification API would give you. Logits stay
private (a probability-only API is also why the paper's l2_logit priming
variant is not reproducible against a real service; see README).
"""

import clip
import torch


class ClipService:
    def __init__(self, class_names, template, device, arch="ViT-B/16"):
        model, preprocess = clip.load(arch, device=device)
        model.float().eval()
        model.requires_grad_(False)
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.n_queries = 0

        with torch.no_grad():
            texts = clip.tokenize([template.format(c) for c in class_names]).to(device)
            emb = model.encode_text(texts)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        self.text_weights = emb.t()  # (d, K)
        self.logit_scale = model.logit_scale.exp().item()

    @torch.no_grad()
    def predict_proba(self, images):
        """images: CLIP-preprocessed batch. Returns (B, K) probabilities."""
        images = images.to(self.device)
        emb = self.model.encode_image(images)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        logits = self.logit_scale * emb @ self.text_weights
        self.n_queries += images.size(0)
        return torch.softmax(logits, dim=-1).cpu()
