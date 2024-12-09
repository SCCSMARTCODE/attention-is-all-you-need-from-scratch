import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, input_vocab_size: int, d_model: int):
        super(InputEmbedding, self).__init__()

        self.input_vocab_size = input_vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=self.input_vocab_size, embedding_dim=d_model)

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)
