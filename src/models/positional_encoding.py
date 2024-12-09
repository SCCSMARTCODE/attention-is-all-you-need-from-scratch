import torch.nn as nn
import math
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(size=(self.seq_length, self.d_model))
        position = torch.arange(start=0, end=self.seq_length, step=1, dtype=torch.float).unsqueeze_(dim=1)
        dev_term = torch.exp(torch.arange(start=0, end=d_model,step=2).float() * (-math.log(10000.0)/d_model))

        pe[:,0::2] = torch.sin(position * dev_term)
        pe[:,1::2] = torch.cos(position * dev_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
