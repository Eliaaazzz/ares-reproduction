"""Zero-shot accuracy of the service model (CLIP ViT-B/16) on the test split.

This is the no-adaptation reference row of Table 2.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ALL_DATASETS, get_dataset, clip_transform
from service import ClipService
from templates import CUSTOM_TEMPLATES
from util import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=ALL_DATASETS, required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_ds, class_names = get_dataset(args.dataset, "test", args.data_dir, None)
    service = ClipService(class_names, CUSTOM_TEMPLATES[args.dataset], device)
    test_ds.transform = clip_transform(service.preprocess, args.dataset)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)

    correct = total = 0
    for x, y in tqdm(loader, ncols=100, desc=f"zero-shot {args.dataset}"):
        probs = service.predict_proba(x)
        correct += (probs.argmax(1) == y).sum().item()
        total += y.size(0)
    acc = correct / total

    save_json(os.path.join(args.results_dir, f"zeroshot_{args.dataset}.json"),
              {"dataset": args.dataset, "test_acc": acc, "n_test": total})
    print(f"{args.dataset}: {100 * acc:.2f}%")


if __name__ == "__main__":
    main()
