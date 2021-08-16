#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
"""
File: source/utils/misc.py
"""

import torch
import argparse
import yaml


class Pack(dict):
    """
    Pack
    """
    def __getattr__(self, name):
        return self.get(name)

    def add(self, **kwargs):
        """
        add
        """
        for k, v in kwargs.items():
            self[k] = v

    def flatten(self):
        """
        flatten
        """
        pack_list = []
        for vs in zip(*self.values()):
            pack = Pack(zip(self.keys(), vs))
            pack_list.append(pack)
        return pack_list

    def cuda(self, device=None):
        """
        cuda
        """
        pack = Pack()
        for k, v in self.items():
            if isinstance(v, tuple):
                pack[k] = tuple(x.cuda(device) for x in v)
            else:
                pack[k] = v.cuda(device)
        return pack


def load_yaml(filename):
    f = open(filename, 'r', encoding='utf-8')
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = Pack(config)
    return config


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(1, *lengths.size(), 1)
    mask = mask.squeeze(0)
    mask = mask.lt(lengths.unsqueeze(-1))
    #mask = mask.repeat(*lengths.size(), 1).lt(lengths.unsqueeze(-1))
    return mask


def max_lens(X):
    """
    max_lens
    """
    if not isinstance(X[0], list):
        return [len(X)]
    elif not isinstance(X[0][0], list):
        return [len(X), max(len(x) for x in X)]
    elif not isinstance(X[0][0][0], list):
        return [len(X), max(len(x) for x in X),
                max(len(x) for xs in X for x in xs)]
    else:
        raise ValueError(
            "Data list whose dim is greater than 3 is not supported!")


def list2tensor(X):
    """
    list2tensor
    """
    size = max_lens(X)
    if len(size) == 1:
        tensor = torch.tensor(X)
        return tensor

    tensor = torch.zeros(size, dtype=torch.long)
    lengths = torch.zeros(size[:-1], dtype=torch.long)
    if len(size) == 2:
        for i, x in enumerate(X):
            l = len(x)
            tensor[i, :l] = torch.tensor(x)
            lengths[i] = l
    else:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                l = len(x)
                tensor[i, j, :l] = torch.tensor(x)
                lengths[i, j] = l

    return tensor, lengths

def one_hot(indice, num_classes):
    """
    one_hot
    """
    I = torch.eye(num_classes).to(indice.device)
    T = I[indice]
    return T


def str2bool(v):
    """
    str2bool
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def sample_gaussian(mu, logvar):
    epsilon = torch.randn(size=logvar.size(), device=logvar.device)
    std = torch.exp((0.5 * logvar))
    z = mu + torch.mul(std, epsilon)

    return z


def gaussian_kld(recognition_mu, recognition_logvar, prior_mu, prior_logvar):
    kld = torch.mul(torch.sum(torch.add((recognition_logvar - prior_logvar)
                     - torch.div(torch.pow(prior_mu - recognition_mu, 2), torch.exp(prior_logvar))
                     - torch.div(torch.exp(recognition_logvar), torch.exp(prior_logvar)), 1), dim=1), -0.5)
    return kld.mean()


def build_copy_mapping(raw_tokens_batch, vocab, unk_token='<unk>', blank_token='<blank>'):
    '''
    raw_tokens_batch list[list]: element is word
    vocab dict
    '''
    new_tokens = set()
    unk_idx = vocab.get(unk_token)
    for raw_tokens in raw_tokens_batch:
        for raw_token in raw_tokens:
            if raw_token == blank_token:
                continue
            if raw_token not in vocab.keys():
                new_tokens.add(raw_token)
    nxt_idx = len(vocab)
    token2idx = dict()
    idx2token = dict()
    for new_token in new_tokens:
        token2idx[new_token] = nxt_idx
        idx2token[nxt_idx] = new_token
        nxt_idx += 1

    def toIdx(w):
        if w in token2idx:
            return token2idx[w]
        return vocab.get(w, unk_idx)
    
    indices = set()
    max_len = 0
    grouped_pos_idx_batch = []
    for raw_tokens in raw_tokens_batch:
        max_len = max(max_len, len(raw_tokens))
        grouped_pos_idx = []
        idx2pos = dict()
        for pos, raw_token in enumerate(raw_tokens):
            idx = toIdx(raw_token)
            idx2pos[idx] = idx2pos.get(idx, [])  + [pos]
            indices.add(idx)
        for idx, pos in idx2pos.items():
            grouped_pos_idx.append((pos, idx))
        grouped_pos_idx_batch.append(grouped_pos_idx)

    indices = list(indices)
    indices.sort()
    idx2idx_mapping = dict(zip(indices, range(len(indices))))
    batch_pos_idx_map = torch.zeros(len(grouped_pos_idx_batch), max_len, len(indices))

    for bidx, grouped_pos_idx in enumerate(grouped_pos_idx_batch):
        for pos, idx in grouped_pos_idx:
            idx = idx2idx_mapping[idx]
            for p in pos:
                batch_pos_idx_map[bidx, p, idx] = 1.
    batch_pos_idx_map = batch_pos_idx_map.cuda()
    
    return token2idx, idx2token, batch_pos_idx_map, idx2idx_mapping



def generate_relative_positions_matrix(length, max_relative_positions,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length+1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t




if __name__ == '__main__':
    X = [1, 2, 3]
    print(X)
    print(list2tensor(X))
    X = [X, [2, 3]]
    print(X)
    print(list2tensor(X))
    X = [X, [[1, 1, 1, 1, 1]]]
    print(X)
    print(list2tensor(X))

    data_list = [{'src': [1, 2, 3], 'tgt': [1, 2, 3, 4]},
                 {'src': [2, 3], 'tgt': [1, 2, 4]}]
    batch = Pack()
    for key in data_list[0].keys():
        batch[key] = list2tensor([x[key] for x in data_list], 8)
    print(batch)
