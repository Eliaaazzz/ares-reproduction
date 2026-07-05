"""Stage 1: Prime once.

Queries the service model a single time per image to get class probabilities,
then distills them into a linear head on top of a frozen local encoder
(KL loss, AdamW, cosine schedule) -- Sec. 3.2 of the paper.

Since the encoder is frozen and the CLIP preprocess is deterministic, the
encoder features are computed once and the 100 distillation epochs run on
cached features. This is mathematically identical to re-forwarding the frozen
encoder every epoch (the official code does the latter) and much faster.

Note on checkpoint selection: the official code picks the head with the lowest
distillation loss on the *test* split, which means the service model is also
queried once per test image. We log train and test query counts separately and
save three heads (best test-KL / best train-KL / last epoch) so the choice can
be ablated downstream.
"""

import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datasets import get_dataset, few_shot_indices, clip_transform, ALL_DATASETS
from models import local_backbone, extract_features
from service import ClipService
from templates import CUSTOM_TEMPLATES
from util import set_seed, save_json


def sweep(loader, service, backbone, device, desc):
    """One pass over the data: query the service API + extract local features."""
    probs, feats, ys = [], [], []
    for x, y in tqdm(loader, desc=desc, ncols=100):
        x = x.to(device)
        probs.append(service.predict_proba(x))
        feats.append(extract_features(backbone, x, device))
        ys.append(y)
    return torch.cat(probs), torch.cat(feats), torch.cat(ys)


def criterion_fn(name):
    if name == "kl":
        return lambda logits, target: F.kl_div(F.log_softmax(logits, dim=1), target, reduction="batchmean")
    if name == "ce":
        return lambda logits, target: F.cross_entropy(logits, target)
    if name == "l2_prob":
        return lambda logits, target: F.mse_loss(F.softmax(logits, dim=1), target)
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=ALL_DATASETS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shots", type=int, default=16)
    p.add_argument("--student", default="vitb16")
    p.add_argument("--criterion", choices=["kl", "ce", "l2_prob"], default="kl")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--cache-dir", default="./cache")
    p.add_argument("--runs-dir", default="./runs")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    tag = f"{args.dataset}_s{args.seed}" + ("" if args.criterion == "kl" else f"_{args.criterion}")

    service_holder = {}

    def get_service(class_names):
        if "s" not in service_holder:
            service_holder["s"] = ClipService(class_names, CUSTOM_TEMPLATES[args.dataset], device)
        return service_holder["s"]

    backbone, feat_dim = local_backbone(args.student)
    backbone.to(device)

    # ---- single-pass interaction with the service API (cached) ----
    os.makedirs(args.cache_dir, exist_ok=True)
    test_cache = os.path.join(args.cache_dir, f"test_{args.dataset}_{args.student}.pt")
    if os.path.exists(test_cache):
        blob = torch.load(test_cache)
        probs_te, feats_te, y_te, class_names = blob["probs"], blob["feats"], blob["ys"], blob["classes"]
        test_queries = 0  # cached from an earlier run
    else:
        # need a throwaway dataset object first to know the class names
        from torch.utils.data import DataLoader
        test_ds, class_names = get_dataset(args.dataset, "test", args.data_dir, None)
        service = get_service(class_names)
        test_ds.transform = clip_transform(service.preprocess, args.dataset)
        loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)
        service.n_queries = 0
        probs_te, feats_te, y_te = sweep(loader, service, backbone, device, "API+features (test)")
        test_queries = service.n_queries
        torch.save({"probs": probs_te, "feats": feats_te, "ys": y_te, "classes": class_names}, test_cache)

    train_cache = os.path.join(args.cache_dir, f"train_{args.dataset}_{args.student}_s{args.seed}_{args.shots}shot.pt")
    if os.path.exists(train_cache):
        blob = torch.load(train_cache)
        probs_tr, feats_tr, y_tr = blob["probs"], blob["feats"], blob["ys"]
        train_queries = 0
    else:
        from torch.utils.data import DataLoader, Subset
        train_ds, class_names = get_dataset(args.dataset, "train", args.data_dir, None)
        service = get_service(class_names)
        train_ds.transform = clip_transform(service.preprocess, args.dataset)
        idx = few_shot_indices(train_ds, args.shots, args.seed)
        loader = DataLoader(Subset(train_ds, idx), batch_size=256, shuffle=False, num_workers=args.num_workers)
        service.n_queries = 0
        probs_tr, feats_tr, y_tr = sweep(loader, service, backbone, device, "API+features (train)")
        train_queries = service.n_queries
        torch.save({"probs": probs_tr, "feats": feats_tr, "ys": y_tr}, train_cache)

    num_classes = len(class_names)
    print(f"{args.dataset}: {len(y_tr)} train ({args.shots}-shot), {len(y_te)} test, {num_classes} classes")

    # ---- distill into the linear head ----
    head = nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    steps_per_epoch = math.ceil(len(y_tr) / args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    loss_fn = criterion_fn(args.criterion)

    feats_tr_d, probs_tr_d = feats_tr.to(device), probs_tr.to(device)
    feats_te_d, probs_te_d = feats_te.to(device), probs_te.to(device)
    y_te_d = y_te.to(device)

    log = {"train_loss": [], "test_loss": [], "test_acc": []}
    best = {"test_kl": (float("inf"), None), "train_kl": (float("inf"), None)}
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(len(y_tr), device=device)
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            sel = perm[i * args.batch_size:(i + 1) * args.batch_size]
            loss = loss_fn(head(feats_tr_d[sel]), probs_tr_d[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            epoch_loss += loss.item() * len(sel)
        train_loss = epoch_loss / len(y_tr)

        head.eval()
        with torch.no_grad():
            logits_te = head(feats_te_d)
            test_loss = loss_fn(logits_te, probs_te_d).item()
            test_acc = (logits_te.argmax(1) == y_te_d).float().mean().item()
        log["train_loss"].append(train_loss)
        log["test_loss"].append(test_loss)
        log["test_acc"].append(test_acc)

        state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if test_loss < best["test_kl"][0]:
            best["test_kl"] = (test_loss, state)
        if train_loss < best["train_kl"][0]:
            best["train_kl"] = (train_loss, state)

    heads = {
        "test_kl": best["test_kl"][1],
        "train_kl": best["train_kl"][1],
        "last": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
    }

    # primed-model accuracy for each selection rule (priming-only baseline)
    accs = {}
    probe = nn.Linear(feat_dim, num_classes).to(device)
    for name, state in heads.items():
        probe.load_state_dict(state)
        with torch.no_grad():
            accs[name] = (probe(feats_te_d).argmax(1) == y_te_d).float().mean().item()

    os.makedirs(args.runs_dir, exist_ok=True)
    ckpt_path = os.path.join(args.runs_dir, f"prime_{tag}.pt")
    torch.save({"heads": heads, "classes": class_names, "args": vars(args)}, ckpt_path)

    summary = {
        "dataset": args.dataset, "seed": args.seed, "shots": args.shots,
        "criterion": args.criterion, "epochs": args.epochs, "lr": args.lr,
        "student": args.student,
        "api_queries_train": train_queries, "api_queries_test": test_queries,
        "primed_test_acc": accs,
        "best_test_kl": best["test_kl"][0],
        "curves": log,
    }
    save_json(os.path.join(args.results_dir, f"prime_{tag}.json"), summary)
    print(f"primed acc: " + ", ".join(f"{k}={v:.4f}" for k, v in accs.items()))
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
