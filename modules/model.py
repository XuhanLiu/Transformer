''' Define the Transformer model '''
import torch
import torch.nn as nn
from torch import optim
from .layer import PositionalEmbedding
from .layer import EncoderLayer, DecoderLayer
from models.optim import ScheduledOptim
import util
from models.generator import Base


def get_pad_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = 1 - torch.triu(torch.ones((1, len_s, len_s)), diagonal=1)
    subsequent_mask = subsequent_mask.bool().to(device=seq.device)
    return subsequent_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, voc, d_emb, n_layers, n_head, d_inner,
                 d_model, pad_idx=0, dropout=0.1):

        super().__init__()

        self.token_emb = nn.Embedding(voc.size, d_emb, padding_idx=pad_idx)
        self.posit_emb = PositionalEmbedding(d_emb, max_len=voc.max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_inner=d_inner, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_emb, eps=1e-6)

    def forward(self, src_seq, src_mask=None):

        # -- Forward

        enc_output = self.dropout(self.posit_emb(src_seq) + self.token_emb(src_seq))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, attn_mask=src_mask)
        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, voc, d_emb, n_layers, n_head, d_inner,
                 d_model, pad_idx=0, dropout=0.1):

        super().__init__()

        self.token_emb = nn.Embedding(voc.size, d_emb, padding_idx=pad_idx)
        self.posit_emb = PositionalEmbedding(d_emb, max_len=voc.max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_inner, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask):
        dec_output = self.dropout(self.posit_emb(trg_seq) + self.token_emb(trg_seq))

        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, enc_output, trg_mask=trg_mask, src_mask=src_mask)
        return dec_output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, voc_src, voc_trg, src_pad=0, trg_pad=0, d_inner=1024,
                 d_emb=256, d_model=256, n_layers=6, n_head=8, dropout=0.1,
                 emb_weight_sharing=True, prj_weight_sharing=False):

        super().__init__()

        self.src_pad, self.trg_pad = src_pad, trg_pad
        self.voc_src, self.voc_trg = voc_src, voc_trg
        self.encoder = Encoder(voc=voc_src, d_emb=d_emb, d_model=d_model,
                               n_layers=n_layers, n_head=n_head, d_inner=d_inner,
                               pad_idx=src_pad, dropout=dropout)

        self.decoder = Decoder(voc=voc_trg, d_emb=d_emb, d_model=d_model,
                               n_layers=n_layers, n_head=n_head, d_inner=d_inner,
                               pad_idx=trg_pad, dropout=dropout)
        self.prj_word = nn.Linear(d_model, voc_trg.size, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.x_logit_scale = 1.
        if prj_weight_sharing:
            self.prj_word.weight = self.decoder.token_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_weight_sharing:
            self.encoder.token_emb.weight = self.decoder.token_emb.weight

        self.optim = ScheduledOptim(
            optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-09), 2.0, d_model)

    def forward(self, src, trg=None):
        src_mask = get_pad_mask(src, self.src_pad)
        memory = self.encoder(src, src_mask)
        if trg is not None:
            trg_mask = get_pad_mask(trg[:, :-1], self.trg_pad) & get_subsequent_mask(trg[:, :-1])
            dec_out = self.decoder(trg[:, :-1], trg_mask, memory, src_mask)
            dec_out = (self.prj_word(dec_out) * self.x_logit_scale).log_softmax(dim=-1)
            output = dec_out.gather(2, trg[:, 1:].unsqueeze(2)).squeeze(2)
        else:
            batch_size, seq_len = src.size()
            output = torch.ones(batch_size, self.voc_trg.max_len).long().to(util.dev)
            # output[:, 0] = torch.LongTensor([self.voc_trg.tk2ix['GO']] * batch_size).to(util.dev)
            isEnd = torch.zeros(batch_size).bool().to(util.dev)

            for step in range(1, self.voc_trg.max_len):  # decode up to max length
                trg_mask = get_pad_mask(output[:, :step]) & get_subsequent_mask(output[:, :step])
                dec_out = self.decoder(output[:, :step], trg_mask, memory, src_mask)
                score = (self.prj_word(dec_out) * self.x_logit_scale).softmax(dim=-1).data
                output[:, step] = torch.multinomial(score[:, step-1, :], 1).view(-1).data
                isEnd |= output[:, step] == self.voc_trg.tk2ix['EOS']
                output[isEnd, step] = self.voc_trg.tk2ix['EOS']
                if isEnd.all(): break
        return output

    def fit(self, pair_loader, ind_loader, epochs=100, out=None):
        log = open(out + '.log', 'w')
        best = 0.
        if len(util.devices) > 1:
            net = nn.DataParallel(self, device_ids=util.devices)
        else:
            net = self
        for epoch in range(epochs):
            for i, (src, trg) in enumerate(pair_loader):
                src, trg = src.to(util.dev), trg.to(util.dev)
                self.optim.zero_grad()
                output = net(src, trg)
                loss = - output.mean()
                print(epoch, i, loss)
                loss.backward()
                self.optim.step()

                if i % 100 != 0 or i == 0: continue
                frags, smiles, valids, desires = [], [], [], []
                for _ in range(1):
                    for ix, src in ind_loader:
                        trg = net(src.to(util.dev))
                        ix = ind_loader.dataset.index[ix]
                        src = [self.voc_src.decode(s) for s in src]
                        trg = [self.voc_src.decode(t) for t in trg]
                for i, smile in enumerate(smiles):
                    print('%s\t%s' % (frags[i], smile), file=log)
        log.close()


