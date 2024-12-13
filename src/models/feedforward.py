from torch import nn
import torch


class PWFeedForwardNetworks(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(PWFeedForwardNetworks, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_ff, out_features=d_model)
        )

    def forward(self, input_: torch.Tensor):
        return self.network(input_)
