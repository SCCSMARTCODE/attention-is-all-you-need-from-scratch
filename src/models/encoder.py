from torch import nn
from multihead_attention import MultiHeadAttention
from feedforward import PWFeedForwardNetworks
from residual_connection import ResidualConnection


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: PWFeedForwardNetworks, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
