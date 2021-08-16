#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#

import os
import torch

from tqdm import tqdm
from source.inputters.field import TextField
from source.inputters.dataset import Dataset


class Corpus(object):
    """
    Corpus
    """

    def __init__(self, config):

        self.data_dir = config.data_dir
        self.data_prefix = config.data_prefix
        self.min_freq = config.min_freq
        self.max_vocab_size = config.max_vocab_size
        self.data_tag = config.data_tag if config.data_tag else self.data_prefix

        prepared_data_file = self.data_prefix + "_" + str(self.max_vocab_size) + \
                                "_" + self.data_tag + ".data.pt"
        prepared_vocab_file = self.data_prefix + "_" + str(self.max_vocab_size) + \
                                "_" + self.data_tag + ".vocab.pt"

        self.prepared_data_file = os.path.join(self.data_dir, prepared_data_file)
        self.prepared_vocab_file = os.path.join(self.data_dir, prepared_vocab_file)
        self.fields = {}
        self.filter_pred = None
        self.sort_fn = None
        self.data = None
        self.Dataset = Dataset

    def load(self, rebuild=False):
        """
        load
        """
        if not (os.path.exists(self.prepared_data_file) and
                os.path.exists(self.prepared_vocab_file)):
            self.build()
        elif rebuild:
            self.build()
        self.load_vocab(self.prepared_vocab_file)
        self.load_data(self.prepared_data_file)

        self.padding_idx = self.TGT.stoi[self.TGT.pad_token]

    def reload(self, data_type='test', prefix=''):
        """
        reload
        """
        data_file = os.path.join(self.data_dir, self.data_prefix + "." + data_type)
        # data_file = os.path.join(self.data_dir, prefix + "." + data_type)
        print("Reload from {}".format(data_file))
        data_raw = self.read_data(data_file, data_type="test")
        data_examples = self.build_examples(data_raw)
        self.data[data_type] = self.Dataset(data_examples)

        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_data(self, prepared_data_file=None):
        """
        load_data
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        data = torch.load(prepared_data_file)
        self.data = {"train": self.Dataset(data['train']),
                     "valid": self.Dataset(data["valid"]),
                     "test": self.Dataset(data["test"])}
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_vocab(self, prepared_vocab_file):
        """
        load_vocab
        """
        prepared_vocab_file = prepared_vocab_file or self.prepared_vocab_file
        print("Loading prepared vocab from {} ...".format(prepared_vocab_file))
        vocab_dict = torch.load(prepared_vocab_file)

        for name, vocab in vocab_dict.items():
            if name in self.fields:
                self.fields[name].load_vocab(vocab)
        print("Vocabulary size of fields:",
              " ".join("{}-{}".format(name.upper(), field.vocab_size)
                       for name, field in self.fields.items()
                       if isinstance(field, TextField)))

    def read_data(self, data_file, data_type=None):
        """
        Returns
        -------
        data: ``List[Dict]``
        """
        raise NotImplementedError

    def build_vocab(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        field_data_dict = {}
        for name in data[0].keys():
            field = self.fields.get(name)
            if isinstance(field, TextField):
                xs = [x[name] for x in data]
                if field not in field_data_dict:
                    field_data_dict[field] = xs
                else:
                    field_data_dict[field] += xs

        vocab_dict = {}
        for name, field in self.fields.items():
            if field in field_data_dict:
                print("Building vocabulary of field {} ...".format(name.upper()))
                if field.vocab_size == 0:
                    field.build_vocab(field_data_dict[field],
                                      min_freq=self.min_freq,
                                      max_size=self.max_vocab_size)
                vocab_dict[name] = field.dump_vocab()
        return vocab_dict

    def build_examples(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        examples = []
        for raw_data in tqdm(data):
            example = {}
            # if len(raw_data) != len(self.fields):
            #     continue
            for name, strings in raw_data.items():
                example[name] = self.fields[name].numericalize(strings)
            examples.append(example)
        if self.sort_fn is not None:
            print("Sorting examples ...")
            examples = self.sort_fn(examples)
        return examples

    def build(self):
        """
        build
        """
        print("Start to build corpus!")
        train_file = os.path.join(self.data_dir, self.data_prefix + ".train")
        valid_file = os.path.join(self.data_dir, self.data_prefix + ".valid")
        test_file = os.path.join(self.data_dir, self.data_prefix + ".test")

        print("Reading data ...")
        train_raw = self.read_data(train_file, data_type="train")
        valid_raw = self.read_data(valid_file, data_type="valid")
        test_raw = self.read_data(test_file, data_type="test")
        vocab = self.build_vocab(train_raw)

        print("Building TRAIN examples ...")
        train_data = self.build_examples(train_raw)
        print("Building VALID examples ...")
        valid_data = self.build_examples(valid_raw)
        print("Building TEST examples ...")
        test_data = self.build_examples(test_raw)

        data = {"train": train_data,
                "valid": valid_data,
                "test": test_data}

        print("Saving prepared vocab ...")
        torch.save(vocab, self.prepared_vocab_file)
        print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))

        print("Saving prepared data ...")
        torch.save(data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))

    def create_batches(self, batch_size, data_type="train",
                       shuffle=False, device=None):
        """
        create_batches
        """
        try:
            data = self.data[data_type]
            data_loader = data.create_batches(batch_size, shuffle, device)
            return data_loader
        except KeyError:
            raise KeyError("Unsupported data type: {}!".format(data_type))

    def transform(self, data_file, batch_size,
                  data_type="test", shuffle=False, device=None):
        """
        Transform raw text from data_file to Dataset and create data loader.
        """
        raw_data = self.read_data(data_file, data_type=data_type)
        examples = self.build_examples(raw_data)
        data = self.Dataset(examples)
        data_loader = data.create_batches(batch_size, shuffle, device)
        return data_loader
