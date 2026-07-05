"""Ablations from the README (all seed 0). Run after run_all.py so the
priming checkpoints exist. Idempotent like the main runner.
"""

import argparse
import os

from run_all import run

# (script, kwargs, results filename to skip on)
JOBS = [
    # released-code protocol: stage 2 on the full training split
    ("reprogram.py", dict(dataset="flowers102", shots=-1, out_tag="_full"), "ares_flowers102_s0_full.json"),
    ("reprogram.py", dict(dataset="dtd", shots=-1, out_tag="_full"), "ares_dtd_s0_full.json"),
    # component analysis / design choices, on eurosat
    ("reprogram.py", dict(dataset="eurosat", no_prime=True, out_tag="_noprime"), "ares_eurosat_s0_noprime.json"),
    ("reprogram.py", dict(dataset="eurosat", norm="clip", out_tag="_clipnorm"), "ares_eurosat_s0_clipnorm.json"),
    ("reprogram.py", dict(dataset="eurosat", mapping="identity", out_tag="_identity"), "ares_eurosat_s0_identity.json"),
    ("reprogram.py", dict(dataset="eurosat", head_select="train_kl", out_tag="_trainkl"), "ares_eurosat_s0_trainkl.json"),
    # priming loss (stage 1 re-runs are cheap thanks to the feature cache)
    ("prime.py", dict(dataset="eurosat", criterion="ce"), "prime_eurosat_s0_ce.json"),
    ("prime.py", dict(dataset="eurosat", criterion="l2_prob"), "prime_eurosat_s0_l2_prob.json"),
    ("reprogram.py", dict(dataset="eurosat", prime_criterion="ce", out_tag="_ce"), "ares_eurosat_s0_ce.json"),
    ("reprogram.py", dict(dataset="eurosat", prime_criterion="l2_prob", out_tag="_l2prob"), "ares_eurosat_s0_l2prob.json"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    args = p.parse_args()
    for script, kw, marker in JOBS:
        if os.path.exists(os.path.join(args.results_dir, marker)):
            print("skip", marker)
            continue
        run(script, seed=0, **kw)


if __name__ == "__main__":
    main()
