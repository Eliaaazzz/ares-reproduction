"""Stage 2: Reprogram locally.

Glass-box visual reprogramming on the primed local model: images are resized
to a small source size, zero-padded to 224x224, and a learnable border prompt
(sigmoid-squashed, ILM-VP style) is added. Labels come from the few-shot
target set; the output space is re-weighted with BLM+ re-estimated every
epoch. Adam lr=0.01, MultiStepLR at 50%/72% of 200 epochs, AMP -- all
following the official configuration.

Protocol note (see README): the released official code runs this stage on the
*full* training split even in the 16-shot VLM pipeline (its --num_samples_per_class
flag only selects the stage-1 checkpoint path). --shots 16 here follows the
paper's stated few-shot protocol; --shots -1 follows the released code.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from datasets import (ALL_DATASETS, SOURCE_SIZE, TRAIN_BATCH, get_dataset,
                      few_shot_indices, vr_transform, materialize)
from mapping import blmp_matrix, apply_mapping
from models import primed_model
from templates import IMAGENET_NORM, CLIP_NORM
from util import set_seed, save_json


class PaddingPrompt(nn.Module):
    """Learnable frame around the (centered, zero-padded) target image."""

    def __init__(self, out_size, source_size, normalize):
        super().__init__()
        self.program = nn.Parameter(torch.zeros(3, out_size, out_size))
        self.l_pad = (out_size - source_size + 1) // 2
        self.r_pad = (out_size - source_size) // 2
        mask = torch.zeros(3, source_size, source_size)
        self.register_buffer("mask", F.pad(mask, [self.l_pad, self.r_pad] * 2, value=1))
        self.normalize = normalize

    def forward(self, x):
        x = F.pad(x, [self.l_pad, self.r_pad] * 2, value=0)
        x = x + torch.sigmoid(self.program) * self.mask
        return self.normalize(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=ALL_DATASETS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shots", type=int, default=16, help="-1 = full training split (matches released code)")
    p.add_argument("--student", default="vitb16")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--mapping", choices=["blmp", "identity"], default="blmp")
    p.add_argument("--head-select", choices=["test_kl", "train_kl", "last"], default="test_kl")
    p.add_argument("--prime-criterion", default="kl", help="which priming run to load")
    p.add_argument("--norm", choices=["imagenet", "clip"], default="imagenet")
    p.add_argument("--eval-every", type=int, default=0, help="0 = pick automatically from test size")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--runs-dir", default="./runs")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-tag", default="")
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    prime_tag = f"{args.dataset}_s{args.seed}" + ("" if args.prime_criterion == "kl" else f"_{args.prime_criterion}")
    ckpt = torch.load(os.path.join(args.runs_dir, f"prime_{prime_tag}.pt"))
    class_names = ckpt["classes"]
    num_classes = len(class_names)
    network = primed_model(args.student, num_classes, ckpt["heads"][args.head_select], device)

    # ---- data ----
    tf = vr_transform(args.dataset)
    train_ds, _ = get_dataset(args.dataset, "train", args.data_dir, tf)
    test_ds, _ = get_dataset(args.dataset, "test", args.data_dir, tf)
    if args.shots > 0:
        # same seed => same 16 images per class as the priming stage
        train_ds = Subset(train_ds, few_shot_indices(train_ds, args.shots, args.seed))
    # decode once; every transform above is deterministic
    if len(train_ds) <= 20000:
        train_ds = materialize(train_ds, num_workers=args.num_workers)
    test_ds = materialize(test_ds, num_workers=args.num_workers)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH[args.dataset], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    print(f"{args.dataset}: {len(train_ds)} train / {len(test_ds)} test, {num_classes} classes")

    norm_stats = IMAGENET_NORM if args.norm == "imagenet" else CLIP_NORM
    normalize = transforms.Normalize(norm_stats["mean"], norm_stats["std"])
    prompt = PaddingPrompt(224, SOURCE_SIZE[args.dataset], normalize).to(device)

    opt = torch.optim.Adam(prompt.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(0.5 * args.epochs), int(0.72 * args.epochs)], gamma=0.1)
    scaler = GradScaler()

    if args.eval_every > 0:
        eval_every = args.eval_every
    else:
        n = len(test_ds)
        eval_every = 5 if n > 10000 else (2 if n > 5000 else 1)

    @torch.no_grad()
    def evaluate(matrix):
        correct = total = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            with autocast("cuda"):
                fx = network(prompt(x))
                if matrix is not None:
                    fx = apply_mapping(fx, matrix)
            correct += (fx.argmax(1) == y).sum().item()
            total += y.size(0)
        return correct / total

    log = {"train_acc": [], "test_acc": {}}
    best_acc, best_epoch = 0.0, -1
    pbar = tqdm(range(args.epochs), ncols=100, desc=f"VR {args.dataset} s{args.seed}")
    for epoch in pbar:
        matrix = None
        if args.mapping == "blmp":
            matrix = blmp_matrix(prompt, network, train_loader, device, num_classes)

        prompt.train()
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with autocast("cuda"):
                fx = network(prompt(x))
                if matrix is not None:
                    fx = apply_mapping(fx, matrix)
                loss = F.cross_entropy(fx, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            correct += (fx.argmax(1) == y).sum().item()
            total += y.size(0)
        sched.step()
        log["train_acc"].append(correct / total)

        if (epoch + 1) % eval_every == 0 or epoch == args.epochs - 1:
            prompt.eval()
            acc = evaluate(matrix)
            log["test_acc"][epoch] = acc
            if acc > best_acc:
                best_acc, best_epoch = acc, epoch
                torch.save({"prompt": prompt.state_dict(), "epoch": epoch, "acc": acc,
                            "args": vars(args)},
                           os.path.join(args.runs_dir, f"vr_{args.dataset}_s{args.seed}{args.out_tag}.pt"))
            pbar.set_postfix_str(f"train {100 * correct / total:.1f}%, test {100 * acc:.2f}%, best {100 * best_acc:.2f}%")

    final_acc = log["test_acc"][args.epochs - 1]
    summary = {
        "dataset": args.dataset, "seed": args.seed, "shots": args.shots,
        "mapping": args.mapping, "head_select": args.head_select,
        "prime_criterion": args.prime_criterion, "norm": args.norm,
        "epochs": args.epochs, "lr": args.lr, "eval_every": eval_every,
        "n_train": len(train_ds), "n_test": len(test_ds),
        "best_test_acc": best_acc, "best_epoch": best_epoch,
        "final_test_acc": final_acc,
        "curves": log,
    }
    name = f"ares_{args.dataset}_s{args.seed}{args.out_tag}.json"
    save_json(os.path.join(args.results_dir, name), summary)
    print(f"best {100 * best_acc:.2f}% (epoch {best_epoch}), final {100 * final_acc:.2f}%")


if __name__ == "__main__":
    main()
