import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, num_feat, hidden_dim):
        super().__init__()
        self.to_query = nn.Linear(num_feat, hidden_dim, bias=False)
        self.to_key = nn.Linear(num_feat, hidden_dim, bias=False)
        self.to_value = nn.Linear(num_feat, hidden_dim, bias=False)
        
    def forward(self, x):
        batch_size, num_item, num_feat = x.shape
        q, k, v = self.to_query(x), self.to_key(x), self.to_value(x)
        # query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        #                      for x in (query, key, value)]
        attention = torch.softmax(torch.matmul(q, torch.transpose(k, 1, 2)) / np.sqrt(num_feat), dim=2)
        return torch.matmul(attention, v), attention


if __name__ == '__main__':
    multihead_attn = nn.MultiheadAttention(embed_dim=64, num_heads=2)
    query = torch.randn(7, 64)
    key = torch.randn(4, 64)
    value = torch.randn(4, 64)
    attn_output, attn_output_weights = multihead_attn(query, key, value, need_weights=True)
    print('official: ', attn_output.shape, attn_output_weights.shape)

    self_attention = SelfAttention(64, 8)
    x = torch.randn(2, 4, 64)
    new_x, attention = self_attention(x)
    print('my: ', new_x.shape, attention.shape)
