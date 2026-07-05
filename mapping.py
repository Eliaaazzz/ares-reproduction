"""Bayesian-guided label mapping (BLM+ from Cai et al., used by AReS).

Builds a probabilistic reweighting matrix between the model's output space and
the target label space from the labeled training set, and is re-estimated at
the start of every epoch with the current visual prompt.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def blmp_matrix(prompt, network, loader, device, num_classes, lap=1, topk_ratio=0.15):
    k = max(1, int(num_classes * topk_ratio))
    probs_list, ys = [], []
    for x, y in loader:
        x = x.to(device)
        logits = network(prompt(x))
        probs = F.softmax(logits, dim=1)
        top_values, top_indices = torch.topk(probs, k, dim=1)
        sparse = torch.zeros_like(probs)
        sparse.scatter_(1, top_indices, top_values)
        probs_list.append(sparse.cpu().float())
        ys.append(y)
    probs = torch.cat(probs_list)
    ys = torch.cat(ys).long()

    # accumulate top-k probability mass per (target class, source class)
    matrix = torch.zeros((len(ys.unique()), probs.size(-1)))
    matrix.scatter_add_(0, ys.view(-1, 1).expand(-1, matrix.size(-1)), probs)
    matrix = matrix.t()  # (source, target)

    # row-wise marginal with Laplace smoothing, then column normalization
    row_sum = matrix.sum(dim=1, keepdim=True) + lap
    matrix = matrix / row_sum
    matrix = matrix / matrix.sum(dim=0)
    return matrix.to(device)


def apply_mapping(logits, matrix):
    return logits @ matrix
