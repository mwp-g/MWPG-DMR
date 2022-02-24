import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
from transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask, Embedding


def layer_norm(x, variance_epsilon=1e-12):
    u = x.mean(-1, keepdim=True)
    s = (x - u).pow(2).mean(-1, keepdim=True)
    x = (x - u) / torch.sqrt(s + variance_epsilon)
    return x

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, sum=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if sum:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class MonoEncoder(nn.Module):
    def __init__(self, vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout):
        super(MonoEncoder, self).__init__()
        self.vocab = vocab
        self.src_embed = Embedding(vocab.size, embed_dim, vocab.padding_idx)
        self.src_pos_embed = SinusoidalPositionalEmbedding(embed_dim)
        self.embed_scale = math.sqrt(embed_dim)
        self.transformer = Transformer(layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.dropout = dropout
    def forward(self, input_ids):
        src_repr = self.embed_scale * self.src_embed(input_ids) + self.src_pos_embed(input_ids)
        src_repr = F.dropout(src_repr, p=self.dropout, training=self.training)
        src_mask = torch.eq(input_ids, self.vocab.padding_idx)
        src_repr = self.transformer(src_repr, self_padding_mask=src_mask)
        return src_repr, src_mask

class MonoEncoder_spc(nn.Module):
    def __init__(self, vocab_eq, vocab_wd, layers, embed_dim, ff_embed_dim, num_heads, dropout):
        super(MonoEncoder_spc, self).__init__()
        self.vocab_eq = vocab_eq
        self.vocab_wd = vocab_wd

        self.eq_embed = Embedding(vocab_eq.size, embed_dim, vocab_eq.padding_idx)
        self.eq_pos_embed = SinusoidalPositionalEmbedding(embed_dim)
        self.wd_embed = Embedding(vocab_wd.size, embed_dim, vocab_wd.padding_idx)
        self.wd_pos_embed = SinusoidalPositionalEmbedding(embed_dim)

        self.embed_scale = math.sqrt(embed_dim)

        # self.eq_transformer = Transformer(layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.eq_RNN = nn.GRU(embed_dim, hidden_size=256, num_layers=2, batch_first=False, bidirectional=True)
        self.wd_transformer = Transformer(layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.dropout = dropout

    def forward(self, input_eq, input_wd):
        eq_repr = self.embed_scale * self.eq_embed(input_eq) + self.eq_pos_embed(input_eq)
        # eq_repr = F.dropout(eq_repr, p=self.dropout, training=self.training)
        eq_mask = torch.eq(input_eq, self.vocab_eq.padding_idx)
        eq_repr = self.eq_RNN(eq_repr)
        eq_repr = eq_repr[0]
        # return eq_repr, eq_mask

        wd_repr = self.embed_scale * self.wd_embed(input_wd) + self.wd_pos_embed(input_wd)
        wd_repr = F.dropout(wd_repr, p=self.dropout, training=self.training)
        wd_mask = torch.eq(input_wd, self.vocab_wd.padding_idx)
        wd_repr = self.wd_transformer(wd_repr, self_padding_mask=wd_mask)
        # return wd_repr, wd_mask

        src_repr = torch.cat((eq_repr, wd_repr), 0)
        src_mask = torch.cat((eq_mask, wd_mask), 0)

        return src_repr, src_mask

class RnnEncoder(nn.Module):
    def __init__(self, vocab, embed_size, hidden_size, num_layers):
        super(RnnEncoder, self).__init__()
        self.vocab = vocab
        self.embed_scale = math.sqrt(embed_size)
        self.src_embed = nn.Embedding(vocab.size, embed_size, vocab.padding_idx)
        # self.lstm = nn.LSTM(embed_size, 512, num_layers, batch_first=False)
        self.gru = nn.GRU(embed_size, hidden_size=256, num_layers=num_layers, batch_first=False, bidirectional=True)
        # self.linear = nn.Linear(hidden_size, vocab.size)

    def forward(self, input_ids):
        # Embed word ids to vectors
        x = self.src_embed(input_ids)
        # Forward propagate LSTM
        out = self.gru(x)
        src = out[0]
        # src = torch.mean(src, 1)
        src_mask = torch.eq(input_ids, self.vocab.padding_idx)
        # Reshape output to (batch_size*sequence_length, hidden_size)
        # out = out.reshape(out.size(0) * out.size(1), out.size(2))
        # Decode hidden states of all time steps
        # out = self.linear(out)
        return src, src_mask