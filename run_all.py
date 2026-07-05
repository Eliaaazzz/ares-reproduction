"""Runs the full pipeline: zero-shot baseline, then prime + reprogram for
every dataset/seed. Idempotent -- runs whose results json already exists are
skipped, so it can be interrupted and restarted.
"""

import argparse
import os
import subprocess
import sys

from datasets import ALL_DATASETS

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, **kw):
    cmd = [sys.executable, os.path.join(HERE, script)]
    for k, v in kw.items():
        flag = f"--{k.replace('_', '-')}"
        if v is True:
            cmd.append(flag)
        else:
            cmd += [flag, str(v)]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--results-dir", default="./results")
    args = p.parse_args()

    def done(name):
        return os.path.exists(os.path.join(args.results_dir, name))

    for ds in args.datasets:
        if not done(f"zeroshot_{ds}.json"):
            run("zeroshot.py", dataset=ds)
        for seed in args.seeds:
            if not done(f"prime_{ds}_s{seed}.json"):
                run("prime.py", dataset=ds, seed=seed)
            if not done(f"ares_{ds}_s{seed}.json"):
                run("reprogram.py", dataset=ds, seed=seed)


if __name__ == "__main__":
    main()
