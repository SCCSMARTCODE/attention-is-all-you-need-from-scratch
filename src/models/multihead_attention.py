import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super(MultiHeadAttention, self).__init__()

        assert d_model % h == 0, f"d_model ({d_model}) is not divisible by h ({h})"

        self.d_model = d_model
        self.d_per_h = d_model//h
        self.heads = h
        self.dropout = dropout

        self.query_trans_matrix = nn.Linear(in_features=d_model, out_features=d_model)
        self.key_trans_matrix = nn.Linear(in_features=d_model, out_features=d_model)
        self.value_trans_matrix = nn.Linear(in_features=d_model, out_features=d_model)
        self.output_trans_matrix = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, q, k, v, mask):
        key = self.key_trans_matrix(k)
        query = self.query_trans_matrix(q)
        value = self.value_trans_matrix(v)

        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_per_h).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_per_h).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_per_h).transpose(1, 2)

        x, attention_score = self.generate_attention(query, key, value, mask, nn.Dropout(self.dropout))
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        return self.output_trans_matrix(x)

    def generate_attention(self, query, key, value, mask, dropout: nn.Dropout):
        attention_scores = (query @ key.reshape(-2, -1)) / math.sqrt(self.d_per_h)
        if mask:
            attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = nn.Softmax(dim=-1)
        if dropout:
            attention_scores = dropout(attention_scores)
        return attention_scores @ value, attention_scores
