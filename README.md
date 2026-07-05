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

*(to be filled in as runs finish)*

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
   This reproduction runs both protocols (`--shots 16` / `--shots -1`) to see
   which one the Table 2 numbers correspond to.
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
