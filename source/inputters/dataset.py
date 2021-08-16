#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
"""
File: source/inputters/dataset.py
"""

import torch
from torch.utils.data import DataLoader

from source.utils.misc import Pack
from source.utils.misc import list2tensor


class Dataset(torch.utils.data.Dataset):
    """
    Dataset
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(device=-1):
        """
        collate_fn
        """
        def collate(data_list):
            """
            collate
            """
            batch = Pack()
            for key in data_list[0].keys():
                batch[key] = list2tensor([x[key] for x in data_list])
            if device >= 0:
                batch = batch.cuda(device=device)
            return batch
        return collate

    def create_batches(self, batch_size=1, shuffle=False, device=-1):
        """
        create_batches
        """
        loader = DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=self.collate_fn(device),
                            pin_memory=False)
        return loader



class CopyDataset(Dataset):
    """
    CopyDataset
    """
    def __init__(self, data, vocab):
        super.__init__(data)
        self.vocab = vocab



    @staticmethod
    def collate_fn(vocab, device=-1):
        """
        collate_fn
        """
        def collate(data_list):
            """
            collate
            ---
            data_list: List[Dict]
            """
            batch = Pack()
            for key in data_list[0].keys():
                batch[key] = list2tensor([x[key] for x in data_list])
            if device >= 0:
                batch = batch.cuda(device=device)

            # copy mechanism prepare
            raw_src = [ x['raw_src'].split() for x in data_list]
            token2idx, idx2token, batch_pos_idx_map, idx2idx_mapping \
                = build_copy_mapping(raw_src, vocab)
            batch['token2idx'] = token2idx
            batch['idx2token'] = idx2token
            batch['batch_pos_idx_map'] = batch_pos_idx_map
            batch['idx2idx_mapping'] = idx2idx_mapping
            batch['output'] = '???'

            return batch
        return collate


    def create_batches(self, batch_size=1, shuffle=False, device=-1):
        """
        create_batches
        """
        loader = DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=self.collate_fn(device),
                            pin_memory=False)
        return loader



class WithBowDataset(torch.utils.data.Dataset):
    """
    WithBowDataset
    """
    def __init__(self, data, bow_vocab_size):
        self.data = data
        self.bow_vocab_size = bow_vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(device=-1, bow_vocab_size=37002):
        """
        collate_fn
        """
        def collate(data_list):
            """
            collate
            """
            batch = Pack()
            for key in data_list[0].keys():
                if key=='topic':
                    continue
                batch[key] = list2tensor([x[key] for x in data_list])

            batch_bow = []
            for x in data_list:
                v = torch.zeros(bow_vocab_size, dtype=torch.float)
                x_bow = x['topic'] # dict
                for w, f in x_bow:
                    v[w] += f
                batch_bow.append(v)
            batch['bow'] = torch.stack(batch_bow)

            if device >= 0:
                batch = batch.cuda(device=device)
            return batch
        return collate

    def create_batches(self, batch_size=1, shuffle=False, device=-1):
        """
        create_batches
        """
        loader = DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=self.collate_fn(device, bow_vocab_size=self.bow_vocab_size),
                            pin_memory=False)
        return loader
