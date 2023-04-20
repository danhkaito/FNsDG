import torch
from typing import *
from torch import nn
from model.SOM.utils import *

class Som(nn.Module):

    def __init__(self, size: Tuple[int, int, int], lr):
        super().__init__()
        self.size = size
        self.lr = torch.tensor(lr)
        self.lr0 = torch.tensor(lr)
        self.dRadius = torch.tensor(max(size[:-1]) / 3.0)
        self.dRadius0 = torch.tensor(max(size[:-1]) / 3.0)
        self.weights = nn.Parameter((torch.randn(self.size)), requires_grad=False)
        self.map_indices = torch.ones(size[0], size[1]).nonzero().view(-1, 2)
        self._recorder = dict()

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Calculates pairwise_dist distance of `a` and `b`."""
        return pairwise_dist(a, b, norm_dist)

    def adjust_hyperparam(self, constlr, constDradius):
        self.lr = (self.lr0*constlr).to("cuda:0")
        self.dRadius = (self.dRadius0*constDradius).to("cuda:0")

    def find_bmus(self, distances: torch.Tensor) -> torch.Tensor:
        """Find BMU for each batch in `distances`."""
        min_idxs = distances.argmin(-1)
        # Distances are flattened, so we need to transform 1d indices into 2d map locations
        return torch.stack([torch.div(min_idxs, self.size[1], rounding_mode='floor'), min_idxs % self.size[1]], dim=1)
    
    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """
        1. Calculate distance between `xb` and `weights`;
        2. Find BMU for each element in `xb`
        3. Return BMUs
        """
        n_features = xb.shape[-1]
        # print("Weight shape: "+ str(self.weights.view(-1, n_features).shape))
        distances = self.distance(xb, self.weights.view(-1, n_features))
        # print("Shape distance" + str(distances.shape))
        bmus = self.find_bmus(distances)
        self._recorder['xb'] = xb.clone()
        self._recorder['bmus'] = bmus
        return bmus
  
    def neigh_fn(self, bmus: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Gaussian neighbourhood function."""
        out_shape = (bmus.shape[0], self.size[0], self.size[1], 1)
        index_dist = pairwise_dist(bmus, self.map_indices, norm_dist)
        return gauss(index_dist, sigma).view(out_shape)

    def backward(self) -> None:
        """
        1. Calculate index-distances of codebook elements and bmus
        2. Calculate neighbourhood scaling on index distances
        3. Update weights
        """
        xb, bmus = self._recorder['xb'], self._recorder['bmus']

        batch_size = xb.shape[0]
        n_features = xb.shape[-1]
        elementwise_diffs = pairwise_dist(xb, self.weights.view(-1, n_features), lambda a, b: a - b).view(batch_size, self.size[0], self.size[1], n_features)
        neighbourhood_mults = self.neigh_fn(bmus, self.dRadius)
        # print(f"Radius {self.dRadius.item()}\n")
        # print(f"Leanring Rate {self.lr.item()}\n")
        self.weights+= ((self.lr * neighbourhood_mults * elementwise_diffs / batch_size).sum(0))




