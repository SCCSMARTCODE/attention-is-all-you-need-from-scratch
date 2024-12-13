import torch.nn as nn
import torch


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, input_: torch.Tensor):
        mean = input_.mean(dim=-1, keepdim=True)
        std = input_.mean(dim=-1, keepdim=True)
        return self.alpha * (input_ - mean)/(std + self.eps) + self.beta
