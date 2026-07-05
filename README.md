# Reproducing "Prime Once, then Reprogram Locally" (AReS, CVPR 2026)

Independent re-implementation of **AReS** from

> Zhang, Cai, Liu, Hamm. *Prime Once, then Reprogram Locally: An Efficient
> Alternative to Black-Box Service Model Adaptation.* CVPR 2026 (Highlight).
> [arXiv:2604.01474](https://arxiv.org/abs/2604.01474) ·
> [official code](https://github.com/yunbeizhang/AReS)

Adapting a closed-box service model (an API that returns class probabilities)
with zeroth-order optimization needs ~10⁸ queries and barely works on modern
APIs. AReS instead queries the service **once per training image**, distills
those probabilities into a linear head on a *local* pre-trained encoder
("prime once"), and then runs standard glass-box visual reprogramming on the
local model ("reprogram locally"). No further API calls are ever needed, at
training or inference time.

I wrote this reproduction from scratch in PyTorch (the official repo was used
to pin down hyperparameters and protocol details, no code copied) and ran it
on a single RTX 4060 Ti (8 GB).

## Scope

The paper's main VLM setting: service model = CLIP ViT-B/16 zero-shot
classifier (CoOp templates), local encoder = ImageNet-pretrained ViT-B/16
(timm), 16 shots per class, seeds 0/1/2. Six of the ten datasets — the ones
that fit my disk and GPU budget:

| | flowers102 | dtd | eurosat | oxfordpets | svhn | gtsrb |
|---|---|---|---|---|---|---|
| classes | 102 | 47 | 10 | 37 | 10 | 43 |
| source | authors' LMDB (CoOp split) | LMDB | LMDB | LMDB | torchvision | torchvision |

Stage 1 (priming): AdamW lr 1e-3, 100 epochs, cosine schedule, batch 64, KL
loss. Stage 2 (reprogramming): padding prompt (sigmoid-squashed border around
a 128×128 or 32×32 center), Adam lr 0.01, 200 epochs, MultiStepLR at
100/144, AMP, BLM+ label mapping (Laplace 1, top-k 15%) re-estimated every
epoch. All following the official configuration.

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The four LMDB datasets come from the OneDrive links in the official repo's
`scripts/download_data.sh` (extract into `./data/<name>/`; note the links
need a cookie jar, e.g. `curl -L -c jar -b jar -o eurosat.zip "<url>"`).
SVHN and GTSRB download themselves via torchvision.

## Run

```bash
python run_all.py                 # zero-shot + prime + reprogram, 6 datasets x 3 seeds
python summarize.py               # aggregate results/ into the tables below
```

Individual stages:

```bash
python zeroshot.py  --dataset eurosat
python prime.py     --dataset eurosat --seed 0
python reprogram.py --dataset eurosat --seed 0
```

## Results

Best-epoch test accuracy, seed 0 (the 3-seed averages are still running and
will replace this table). Paper numbers are the corresponding columns of
Table 2.

| Method | flowers102 | dtd | eurosat | oxfordpets | svhn | gtsrb | Avg |
|---|---|---|---|---|---|---|---|
| Zero-shot CLIP (mine) | 70.8 | 43.9 | 48.3 | 89.0 | 19.9 | 21.0 | 48.8 |
| Zero-shot CLIP (paper) | 71.3 | 43.9 | 47.9 | 89.1 | 17.9 | 21.0 | 48.7 |
| Priming only (mine) | 72.8 | 39.1 | 41.2 | 86.9 | 8.7 | 10.4 | 43.2 |
| **AReS, 16-shot as stated in the paper (mine)** | 80.6 | 43.9 | 76.2 | 80.5 | 19.9 | 18.1 | 53.2 |
| AReS final-epoch instead of best (mine) | 79.0 | 40.4 | 74.2 | 77.7 | 17.4 | 17.9 | 51.1 |
| AReS (paper Table 2) | 86.6 | 48.2 | 85.7 | 88.9 | 63.2 | 39.4 | 67.0 |
| BlackVIP (paper Table 2) | 70.6 | 45.3 | 73.3 | 89.1 | 44.4 | 21.3 | 57.3 |

The zero-shot row matching the paper (±0.5, ±2.0 on svhn) says the service
model, templates and splits are faithful. The 16-shot AReS row does not match
Table 2 — which brings us to the protocol question.

### Which protocol produced Table 2?

Running stage 2 the way the *released code* does (full training split,
`--shots -1`) instead of the *paper's stated* 16 shots:

| flowers102 | stage-2 images | best acc |
|---|---|---|
| 16-shot (paper protocol) | 1,632 | 80.6 |
| full split (released code) | 4,093 | **86.6** &nbsp;*(paper: 86.6)* |

| dtd | stage-2 images | best acc |
|---|---|---|
| 16-shot (paper protocol) | 752 | 43.9 |
| full split (released code) | 2,820 | **52.4** &nbsp;*(paper: 48.2)* |

On flowers102 the full-split run reproduces the paper's number exactly to one
decimal; the strict 16-shot run is 6 points below. The gap also scales with
how much bigger the full split is: svhn's full split is 458× its 16-shot set
(73,257 vs 160 images) and shows the largest gap (19.9 vs 63.2), while
flowers102's is only 2.5× and shows the smallest. My best guess is that the
Table 2 VLM numbers were produced with the released code's full-split stage 2,
with only the priming stage actually few-shot. Happy to be corrected if the
16-shot claim refers to something else.

Two related observations:

* Where the service teacher is near-random (svhn 17.9%, gtsrb 21.0%
  zero-shot), priming distills a near-random head (8.7% / 10.4%), and 16
  labeled shots per class cannot climb out of that hole — those two paper
  numbers seem unreachable without the full label set.
* Where the teacher is strong, the opposite failure appears: on oxfordpets the
  16-shot prompt *hurts* (priming-only 86.9 → AReS 80.5), i.e. the visual
  prompt overfits 592 images.

### Ablations (eurosat, seed 0, best epoch)

| Variant | Acc |
|---|---|
| AReS default (KL prime, test-KL head, BLM+, ImageNet norm) | 76.2 |
| reprogram the raw ImageNet ViT instead (no priming) | 75.9 |
| identity mapping instead of BLM+ | 77.9 |
| head selected on train-KL instead of test-KL | **80.6** |
| CLIP normalization in stage 2 (aligned with stage 1) | 70.9 |
| priming loss CE instead of KL | 68.0 |
| priming loss L2-prob instead of KL | 68.2 |

* KL is clearly the best priming loss downstream (matches the paper's Fig.
  3c) — interestingly its primed-only accuracy is *lower* than CE/L2 (41.2 vs
  43.5/43.8), so raw distillation accuracy is not what makes a head a good
  reprogramming substrate.
* Selecting the priming head on the few-shot train loss — the strictly legal
  choice — beats the released code's test-split selection by 4.4 points, so
  the test-set quirk is not even buying accuracy.
* In the strict 16-shot regime priming contributes ~nothing on eurosat (75.9
  without it); its value shows up when the teacher is strong (pets 86.9
  priming-only). The +15-point priming effect in the paper's Table 6 likely
  also reflects the full-split protocol.
* With aligned label spaces, identity mapping edges out BLM+ — consistent
  with the paper's text (which says VLMs use identity), though the released
  VLM scripts pass BLM+.
* Keeping the stage-1/stage-2 normalization mismatch of the official code is
  *better* than aligning both to CLIP stats: the ImageNet statistics match
  the local encoder's pretraining, and the prompt absorbs the head's shift.

Priming query counts (train split, one query per image): 1,632 / 752 / 160 /
592 / 160 / 688 for the six datasets — "prime once" is indeed cheap. The
official checkpoint selection additionally queries each test image once
(2,463 / 1,692 / 8,100 / 3,669 / 26,032 / 12,630), which the train-KL variant
above avoids entirely.

## Observations on the official code

Reading the released code closely turned up a few things that were useful to
know when trying to match the paper's numbers. Flagging them here mostly as
questions, not criticism:

1. **Stage 2 trains on the full split, not 16-shot.** The released
   `reprogram.py` loads its data through `prepare_padding_data`, which returns
   the *full* training split; its `--num_samples_per_class 16` argument only
   selects the stage-1 checkpoint path. So in the released pipeline only the
   priming stage is few-shot, while the visual prompt and the BLM+ mapping are
   fit on all labels (e.g. 4,093 images for Flowers102 instead of 1,632).
   `prepare_padding_data_few_shot` exists in the repo but is never called.
   This reproduction runs both protocols (`--shots 16` / `--shots -1`); the
   results above are consistent with Table 2 coming from the full-split
   variant.
2. **Checkpoint selection touches the test split.** The priming stage picks
   the head with the lowest distillation loss on the *test* set (so the
   service model is also queried once per test image — which the paper's API
   accounting appears to include), and due to how the loss variable is read
   after the eval loop, the selection effectively uses the last test batch
   only. I save three heads (best test-KL / best train-KL / last epoch) and
   ablate the choice.
3. **Normalization changes between stages.** Priming feeds the local encoder
   CLIP-normalized 224² images; reprogramming feeds ImageNet-normalized
   padded images. The frozen head trained in stage 1 therefore sees a shifted
   input distribution in stage 2 and the prompt has to absorb the difference.
   `--norm clip` ablates this.
4. **Best-epoch reporting.** The official script tracks the best test accuracy
   across all 200 epochs (standard in the VR literature, but it is test-set
   model selection). I report both best and final epoch.
5. The `l2_logit` priming criterion in the official code requires raw logits,
   which a probability-returning API does not expose — consistent with the
   paper's own framing, only prob-based losses (`kl`, `ce`, `l2_prob`) are
   reproducible against a real service.

## Implementation differences

* Stage 1 extracts the frozen encoder's features once and trains the head on
  the cache. With a deterministic preprocess and a LayerNorm-only ViT this is
  mathematically identical to re-forwarding the encoder every epoch, and cuts
  stage 1 from minutes to seconds.
* Stage 2 uses size-weighted gradient accumulation (micro-batch 64) so the
  parameter updates match the official batch-256 optimization on an 8 GB
  card, and evaluates every 1/2/5 epochs depending on test-set size rather
  than every epoch.
* Windows portability: lmdb environments open lazily and transforms avoid
  lambdas so datasets survive `spawn`-based DataLoader workers.

## Acknowledgements

Datasets and splits follow [ILM-VP](https://github.com/OPTML-Group/ILM-VP)
(CoOp splits); the BLM+ mapping follows
[BayesianLM](https://github.com/tmlr-group/BayesianLM). Both are also the
basis of the official AReS pipeline.
