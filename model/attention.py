import torch
import torch.nn as nn
import numpy as np


class MultiHeadAttention(nn.Module):
    '''
    essential: change the dimension of input feature (query) from dim_key to dim_value
    however: we believe query get some information from reply because of dot product
    '''
    def __init__(self, in_channels, out_channels, dim_reply, dim_key, dim_value, num_head, dropout=None):
        super(MultiHeadAttention, self).__init__()
        # for multihead computation, dim_key and dim_value should be divided by num_head
        assert dim_key % num_head == 0
        assert dim_value % num_head == 0

        self.num_head = num_head
        self.d_k = dim_key // num_head
        self.d_v = dim_value // num_head
        self.use_dropout = False
        if dropout is not None:
            self.use_dropout = True
            self.dropout = nn.Dropout()

        # match dim of query and key
        # in my opinion, no need for this if query has same dim as key
        self.to_query = nn.Linear(in_channels, dim_key)
        self.to_key = nn.Linear(dim_reply, dim_key)
        self.to_value = nn.Linear(dim_reply, dim_value)
        # aggregate info of all heads
        self.to_out = nn.Linear(dim_value, out_channels)

    def attention(self, query, key, value, mask=None):
        '''
        return:

        note:
            Scaled Dot Product Attention
        '''
        dot_prod = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            dot_prod = dot_prod.masked_fill_(mask==0, -1e9)
        weight = torch.softmax(dot_prod, -1)    # normalize across value in single query
        if self.use_dropout:
            weight = self.dropout(weight)
        return torch.matmul(weight, value), weight

    def forward(self, mismatch_query, reply, mask=None):
        '''
        input:
            mismatch_query: batch, item_query, dim_input
            reply:          batch, item_reply, dim_reply
            mask:           batch, item_query, item_reply

        return:
            better represented query:   batch, item_query, dim_value
            weight:                     batch, item_query, item_reply

        note:
            query must have its last dim same as dim_key (for dot product)
            item_query and item_reply can be different. e.g. translation, QA
            weight shows how much attention query pays to key
        '''
        # same mask for all heads
        batch_size = mismatch_query.shape[0]
        if mask is not None:
            mask = mask.unsqueeze(1)    # batch, 1, item_q, item_r

        query, key, value = self.to_query(mismatch_query), self.to_key(reply), self.to_value(reply)
        query = query.view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)     # batch, head, item_q, d_k
        key = key.view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)         # batch, head, item_r, d_k
        value = value.view(batch_size, -1, self.num_head, self.d_v).transpose(1, 2)     # batch, head, item_r, d_v

        # batch, head, item_q, d_v;     batch, head, item_q, item_r
        weighted_value, weight = self.attention(query, key, value, mask=mask)
        # concat by 'view' into: batch, item_q, dim_value
        weighted_value = weighted_value.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.d_v)
        # mean across heads: batch, item_q, item_r
        weight = torch.mean(weight, dim=1)

        return self.to_out(weighted_value), weight


if __name__ == '__main__':
    # query: are you a good man
    # reply: 我不是
    bs = 10

    # official: use same feature dim for qkv, which forces input and output same shape
    multihead_attn = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
    query = torch.randn(bs, 5, 64)
    key = torch.randn(bs, 3, 64)
    value = torch.randn(bs, 3, 64)
    attn_output, attn_output_weights = multihead_attn(query, key, value, need_weights=True)
    print('official: ', attn_output.shape, attn_output_weights.shape)
    # torch.Size([10, 5, 64]) torch.Size([10, 5, 3])

    # mine: transform query embedding dim into a needed dim with the info of reply
    needed = 16
    english, chinese = 32, 64
    my_query, my_reply = torch.randn(bs, 5, english), torch.randn(bs, 3, chinese)
    self_attention = MultiHeadAttention(
        in_channels=english, 
        out_channels=needed, 
        dim_reply=chinese, 
        dim_key=4, 
        dim_value=8, 
        num_head=2,
    )
    attention, weight = self_attention(my_query, my_reply)
    print('mine: ', attention.shape, weight.shape)
    # torch.Size([10, 5, 16]) torch.Size([10, 5, 3])
