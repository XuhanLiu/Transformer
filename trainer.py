#!/usr/bin/env python
import torch
from rdkit import rdBase
from models import generator
import util
import pandas as pd
from modules import Transformer
from torch.utils.data import DataLoader

rdBase.DisableLog('rdApp.error')
torch.set_num_threads(1)
BATCH_SIZE = 16


def pretrain():
    voc = util.VocCmp('data/voc.txt', max_len=101)
    agent_path = 'output/netAttn_%d' % (BATCH_SIZE)
    agent = Transformer(voc, voc).to(util.dev)
    df = pd.read_table("data/ar_pair.txt")
    test = df.sample(len(df) // 5)
    data = df.drop(test.index)
    input = [voc.encode(('GO ' + seq).split(' ')) for seq in data.INPUT.values]
    output = [voc.encode(('GO ' + seq).split(' ')) for seq in data.OUTPUT.values]
    ind_set = [voc.encode(('GO ' + seq).split(' ')) for seq in set(test.INPUT.values)]
    ind_set = util.TgtData(ind_set, [voc.decode(seq) for seq in ind_set], max_len=voc.max_len)
    ind_loader = DataLoader(ind_set, batch_size=BATCH_SIZE, collate_fn=ind_set.collate_fn)
    pair_set = util.PairData(input, output, src_len=101, trg_len=101)
    pair_loader = DataLoader(pair_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pair_set.collate_fn)
    agent.fit(pair_loader, ind_loader, epochs=1000, out=agent_path)


if __name__ == "__main__":
    # pair = ['inchi_key', 'accession', 'pchembl_value']
    pair = ['cmp_id', 'tgt_id', 'pchembl_value']
    # DrugSeq2Seq(is_attn=True)
    pretrain()