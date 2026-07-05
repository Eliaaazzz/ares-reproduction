"""Local model (the "amenable" encoder that gets primed)."""

import timm
import torch
import torch.nn as nn

TIMM_NAMES = {
    "vitb16": "vit_base_patch16_224",
    "vitb32": "vit_base_patch32_224",
}


def local_backbone(name="vitb16"):
    """ImageNet-pretrained ViT from timm, head removed."""
    model = timm.create_model(TIMM_NAMES[name], pretrained=True)
    feat_dim = model.head.in_features
    model.head = nn.Identity()
    model.eval()
    model.requires_grad_(False)
    return model, feat_dim


@torch.no_grad()
def extract_features(model, images, device):
    """Head input features for a batch (pre_logits so it matches model.head)."""
    feats = model.forward_features(images.to(device))
    return model.forward_head(feats, pre_logits=True).cpu()


def primed_model(name, num_classes, head_state, device):
    """Backbone + the linear head trained in the priming stage, frozen."""
    model = timm.create_model(TIMM_NAMES[name], pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    model.head.load_state_dict(head_state)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model
