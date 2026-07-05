"""Aggregates results/*.json into the markdown tables used in the README."""

import argparse
import glob
import json
import os
import statistics as st

from datasets import ALL_DATASETS


def collect(results_dir):
    out = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            out[os.path.basename(path)[:-5]] = json.load(f)
    return out


def fmt_mean_std(values):
    if not values:
        return "-"
    if len(values) == 1:
        return f"{100 * values[0]:.1f}"
    return f"{100 * st.mean(values):.1f} ± {100 * st.stdev(values):.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()
    r = collect(args.results_dir)
    datasets = [ds for ds in ALL_DATASETS if f"zeroshot_{ds}" in r or any(f"ares_{ds}_s{s}" in r for s in args.seeds)]

    rows = []

    def row(label, per_ds):
        cells = [per_ds(ds) for ds in datasets]
        means = []
        for ds in datasets:
            v = per_ds(ds, raw=True)
            if v:
                means.append(100 * st.mean(v))
        avg = f"{st.mean(means):.1f}" if len(means) == len(datasets) else "-"
        rows.append(f"| {label} | " + " | ".join(cells) + f" | {avg} |")

    def getter(prefix, key, per_seed=True, sub=None):
        def g(ds, raw=False):
            vals = []
            names = [f"{prefix}_{ds}_s{s}" for s in args.seeds] if per_seed else [f"{prefix}_{ds}"]
            for n in names:
                if n in r:
                    v = r[n][key]
                    vals.append(v[sub] if sub else v)
            return vals if raw else fmt_mean_std(vals)
        return g

    header = "| Method | " + " | ".join(datasets) + " | Avg |"
    sep = "|---" * (len(datasets) + 2) + "|"
    row("Zero-shot CLIP (service)", getter("zeroshot", "test_acc", per_seed=False))
    row("Priming only", getter("prime", "primed_test_acc", sub="test_kl"))
    row("AReS (best epoch)", getter("ares", "best_test_acc"))
    row("AReS (final epoch)", getter("ares", "final_test_acc"))

    print(header)
    print(sep)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
