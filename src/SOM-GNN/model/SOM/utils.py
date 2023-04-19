import torch
from typing import *
import numpy as np


def pairwise_dist(a: torch.Tensor, b: torch.Tensor, fn: Callable) -> torch.Tensor:
    """
    Makes `a` and `b` shapes compatible, then calls `fn(a, b)`.
    """
    N, M = a.shape[0], b.shape[0]
    # Reshape a, expand b to match shape
    _a = a.view(N, 1, -1).to(device=a.device)
    _b = b.expand(N, -1, -1).to(device=a.device)
    
    # Invoke `fn` on pairwise_dist a and b
    res = fn(_a, _b)
    
    return res


def norm_dist(a: torch.Tensor, b: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Calculates distance of order `p` between `a` and `b`.
    """
    return (a-b).abs().pow(p).sum(-1).pow(1/p)

def gauss(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Calculates the gaussian of `x` with `radius`."""
    return torch.exp(torch.neg(x.pow(2) / (2*radius.pow(2))))


def idxs_2d_to_1d(idxs: np.ndarray, row_size: int) -> list:
    """Transforms an `np.ndarray` of indices from 2D to 1D by using `row_size`."""
    return torch.tensor([el[0] * row_size + el[1] for el in idxs])
  
def mean_quantization_err(pred_b: torch.Tensor, som = None) -> torch.Tensor:
    """Mean distance of each record from its respective BMU."""
    xb = som._recorder['xb']
    w = som.weights.view(-1, xb.shape[-1])
    row_sz = som.size[0]
    preds = idxs_2d_to_1d(pred_b, row_sz)
    return norm_dist(xb, w[preds], p=2).mean()

