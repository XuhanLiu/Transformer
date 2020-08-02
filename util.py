import torch
from torch.utils.data import Dataset
import re
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
import os
import joblib
from scipy import linalg
from rdkit import rdBase

torch.set_num_threads(1)
rdBase.DisableLog('rdApp.error')
AA = 'ARNDCQEGHILKMFPSTWYV'
dev = torch.device('cuda')
torch.cuda.set_device(0)
devices = [0]


class VocTgt:
    def __init__(self, max_len=1000):
        self.chars = ['-'] + [r for r in AA]
        self.size = len(self.chars)
        self.max_len = max_len
        self.tk2ix = dict(zip(self.chars, range(len(self.chars))))

    def encode(self, seq):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = torch.zeros(self.max_len)
        for i in range(len(seq)):
            res = seq[i] if seq[i] in self.chars else '-'
            smiles_matrix[i] = self.tk2ix[res]
        return smiles_matrix


class VocCmp:
    """A class for handling encoding/decoding from SMILES to an array of indices"""

    def __init__(self, init_from_file=None, max_len=100):
        self.chars = ['EOS', 'GO']
        if init_from_file: self.init_from_file(init_from_file)
        self.size = len(self.chars)
        self.tk2ix = dict(zip(self.chars, range(len(self.chars))))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}
        self.max_len = max_len

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.long)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.tk2ix[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i.item() == self.tk2ix['GO']: continue
            if i.item() == self.tk2ix['EOS']: break
            chars.append(self.ix2tk[i.item()])
        smiles = "".join(chars)
        smiles = smiles.replace('L', 'Cl').replace('R', 'Br')
        return smiles

    def tokenize(self, smile):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,6}\])'
        smile = smile.replace('Cl', 'L').replace('Br', 'R')
        tokens = []
        for word in re.split(regex, smile):
            if word == '' or word is None: continue
            if word.startswith('['):
                tokens.append(word)
            else:
                for i, char in enumerate(word):
                    tokens.append(char)
        return tokens

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split() + ['*']
            self.chars += sorted(set(chars))


class TgtData(Dataset):
    def __init__(self, seqs, ix, max_len=100):
        self.max_len = max_len
        self.index = np.array(ix)
        self.map = {idx: i for i, idx in enumerate(self.index)}
        self.seq = seqs

    def __getitem__(self, i):
        seq = self.seq[i]
        return i, torch.LongTensor(seq)

    def __len__(self):
        return len(self.seq)

    def collate_fn(self, arr):
        collated_ix = np.zeros(len(arr), dtype=int)
        collated_seq = torch.zeros(len(arr), self.max_len).long()
        for i, (ix, tgt) in enumerate(arr):
            collated_ix[i] = ix
            collated_seq[i, :tgt.size(0)] = tgt
        return collated_ix, collated_seq


class SeqData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""

    def __init__(self, input):
        self.input = input

    def __getitem__(self, i):
        return self.input[i]

    def __len__(self):
        return len(self.input)

    @classmethod
    def collate_fn(cls, arr, max_len=100):
        """Function to take a list of encoded sequences and turn them into a batch"""
        # max_length = max([seq.size(0) for seq in arr])
        collated_arr = torch.zeros(len(arr), max_len).long()
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr


class QSARData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""

    def __init__(self, smiles, labels, voc, is_token=False):
        self.voc = voc
        self.labels = labels
        self.smiles = smiles
        self.tokens = []
        for smile in self.smiles:
            token = smile.split(' ') if is_token else self.voc.tokenize(smile)
            if len(token) > self.voc.max_len: continue
            self.tokens.append(token)

    def __getitem__(self, i):
        encoded = self.voc.encode(self.tokens[i])
        return encoded, self.labels[i]

    def __len__(self):
        return len(self.tokens)

    def collate_fn(self, arr, max_len=100):
        """Function to take a list of encoded sequences and turn them into a batch"""
        smiles_arr = torch.zeros(len(arr), self.voc.max_len).long()
        labels_arr = torch.zeros(len(arr), self.labels.shape[1])
        for i, (smile, label) in enumerate(arr):
            smiles_arr[i, :smile.size(0)] = smile
            labels_arr[i, :] = torch.tensor(label)
        return smiles_arr, labels_arr


class PairData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""

    def __init__(self, src, trg, src_len=100, trg_len=100):
        self.src = src
        self.trg = trg
        self.src_len = src_len
        self.trg_len = trg_len
        assert len(self.src) == len(self.trg)

    def __getitem__(self, i):
        src = self.src[i]
        trg = self.trg[i]
        return torch.LongTensor(src), torch.LongTensor(trg)

    def __len__(self):
        return len(self.src)

    def collate_fn(self, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        # max_length = max([seq.size(0) for seq in arr])
        collated_src = torch.zeros(len(arr), self.src_len).long()
        collated_trg = torch.zeros(len(arr), self.trg_len).long()
        for i, (src, trg) in enumerate(arr):
            collated_src[i, :src.size(0)] = src
            collated_trg[i, :trg.size(0)] = trg
        return collated_src, collated_trg


class PCMData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""

    def __init__(self, voc_tgt, voc_cmp, df=None, token=None):
        self.voc_tgt = voc_tgt
        self.voc_cmp = voc_cmp
        if token is not None:
            if isinstance(df, str) and os.path.exists(df):
                df = pd.read_table(df)
            self.prots = [self.voc_tgt.encode(self.voc_tgt.tokenize(seq)) for seq in df.SEQUENCE.values]
            self.labels = torch.Tensor((df['PCHEMBL_VALUE'] >= 6.5).astype(float))
            if token is not None:
                self.smiles = [self.voc_cmp.encode(tokens.split(' ')) for tokens in df.TOKEN.values]
            else:
                self.smiles = [self.voc_cmp.encode(self.voc_cmp.tokenize(smile)) for smile in
                               df.CANONICAL_SMILES.values]

    def __getitem__(self, i):
        smile = self.smiles[i]
        prot = self.prots[i]
        return prot, smile, self.labels[i]

    def __len__(self):
        return len(self.smiles)

    @classmethod
    def collate_fn(cls, arr, max_cmp=100, max_tgt=1000):
        """Function to take a list of encoded sequences and turn them into a batch"""
        collated_tgt = torch.zeros(len(arr), max_tgt).long()
        collated_cmp = torch.zeros(len(arr), max_cmp).long()
        label_arr = torch.zeros(len(arr), 1)
        for i, (tgt, cmp, label) in enumerate(arr):
            collated_tgt[i, :tgt.size(0)] = tgt
            collated_cmp[i, :cmp.size(0)] = cmp
            label_arr[i, :] = label
        return collated_tgt, collated_cmp, label_arr


def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    return torch.LongTensor(np.sort(idxs)).to(dev)


def check_smiles(seqs, voc=None, frags=None):
    smiles = []
    valids = []
    desires = []
    for j, seq in enumerate(seqs):
        if voc is not None:
            seq = voc.decode(seq)
        try:
            mol = Chem.MolFromSmiles(seq)
            valids.append(0 if mol is None else 1)
        except:
            valids.append(0)
        if frags is not None:
            try:
                frag = Chem.MolFromSmarts(frags[j])
                desires.append(1 if mol.HasSubstructMatch(frag) else 0)
            except:
                desires.append(0)
        smiles.append(seq)

    if frags is None:
        return smiles, valids
    else:
        return smiles, valids, desires


class Environment:
    def __init__(self, clf_paths, objs, radius=3, bit_len=2048, is_reg=False, threshold=None):
        self.clf_paths = clf_paths
        self.clfs = {key: joblib.load(value) for key, value in clf_paths.items()}
        self.radius = radius
        self.bit_len = bit_len
        self.is_reg = is_reg
        self.objs = {} if objs is None else objs
        self.threshold = threshold
        if threshold is None:
            self.threshold = 6.5 if is_reg else 0.5

    def __call__(self, smiles, frags=None):
        fps = self.ECFP_from_SMILES(smiles)
        preds = {}
        for key, clf in self.clfs.items():
            off_target = key in self.objs and self.objs[key] == -1
            if self.is_reg:
                score = clf.predict(fps)
                preds[key] = 13 - score if off_target else score
            else:
                score = clf.predict_proba(fps)
                preds[key] = score[:, 0] if off_target else score[:, 1]
        preds = pd.DataFrame(preds, index=fps.index)

        if frags is not None:
            is_match = []
            for i, frag in enumerate(frags):
                frag = Chem.MolFromSmiles(frag)
                mol = Chem.MolFromSmiles(smiles[i])
                if mol is None:
                    is_match.append(0)
                else:
                    # is_match.append(1 if mol.HasSubstructMatch(frag) else 0)
                    fp_mol = AllChem.GetMorganFingerprint(mol, 3)
                    fp_frag = AllChem.GetMorganFingerprint(frag, 3)
                    is_match += [DataStructs.TverskySimilarity(fp_frag, fp_mol, 1, 0)]
            preds['MATCH'] = is_match
            preds['DESIRE'] = ((preds >= self.threshold).sum(axis=1) == len(self.clfs) + 1)
        else:
            preds['DESIRE'] = ((preds >= self.threshold).sum(axis=1) == len(self.clfs))
        preds['VALID'] = (fps.sum(axis=1) > 0).astype(int)
        preds[preds.VALID == 0] = 0
        return preds

    @classmethod
    def ECFP_from_SMILES(cls, smiles, radius=3, bit_len=2048, scaffold=0, index=None):
        fps = np.zeros((len(smiles), bit_len))
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            arr = np.zeros((1,))
            try:
                if scaffold == 1:
                    mol = MurckoScaffold.GetScaffoldForMol(mol)
                elif scaffold == 2:
                    mol = MurckoScaffold.MakeScaffoldGeneric(mol)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps[i, :] = arr
            except:
                print(smile)
                fps[i, :] = [0] * bit_len
        return pd.DataFrame(fps, index=(smiles if index is None else index))

    @classmethod
    def calculate_frechet_distance(cls, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representive data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    @classmethod
    def calc_ffd(cls, smiles1, smiles2):
        fps1 = cls.ECFP_from_SMILES(smiles1)
        mu1 = np.mean(fps1, axis=0)
        sigma1 = np.cov(fps1, rowvar=False)

        fps2 = cls.ECFP_from_SMILES(smiles2)
        mu2 = np.mean(fps2, axis=0)
        sigma2 = np.cov(fps2, rowvar=False)
        ffd = cls.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return ffd

    @classmethod
    def calc_fid(cls, model, smiles1, smiles2):
        fps1 = cls.ECFP_from_SMILES(smiles1)
        act1 = model.inception(fps1)
        mu1 = np.mean(act1, axis=0)
        sigma1 = np.cov(act1, rowvar=False)

        fps2 = cls.ECFP_from_SMILES(smiles2)
        act2 = model.inception(fps2)
        mu2 = np.mean(act2, axis=0)
        sigma2 = np.cov(act2, rowvar=False)
        fid = cls.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid

# class SmilesWraper(Converter):
#     def __init__(self, branches=True, rings=True):
#         super(SmilesWraper, self).__init__(branches=branches, rings=rings)
#
#     def encode(self, smiles):
#         smiles = super(SmilesWraper, self).encode(smiles)
#
#         def count(matched):
#             size = len(matched.group())
#             return ')%d' % size
#
#         return re.sub('\)\)+', count, smiles)
#
#     def decode(self, deepsmiles):
#         def count(matched):
#             size = int(matched.group()[1:])
#             return ')' * size
#
#         deepsmiles = re.sub('\)\d+', count, deepsmiles)
#         smiles = super(SmilesWraper, self).decode(deepsmiles)
#         return smiles
