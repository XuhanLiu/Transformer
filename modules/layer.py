''' Define the Layers '''
import torch
import torch.nn as nn
from .attn import MultiHeadAttention, PositionwiseFeedForward
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=100):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].detach()


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, n_head, d_inner, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.pffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, attn_mask=None):
        enc_output = self.attn(enc_input, enc_input, enc_input, mask=attn_mask)
        enc_output = self.pffn(enc_output)
        return enc_output


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, n_head, d_inner, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, trg_mask=None, src_mask=None):
        dec_output = self.slf_attn(dec_input, dec_input, dec_input, mask=trg_mask)
        dec_output = self.enc_attn(dec_output, enc_output, enc_output, mask=src_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output