import torch.nn as nn
import math
import torch


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, input: torch.Tensor):
        mean = input.mean(dim=-1, keepdim=True)
        std = input.mean(dim=-1, keepdim=True)
        return self.alpha * (input - mean)/(std + self.eps) + self.beta
