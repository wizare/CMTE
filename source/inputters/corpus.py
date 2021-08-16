#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
import os
import json
import random

from source.inputters.base_corpus import Corpus
from source.inputters.tokenizer import simp_zh_tokenize as tokenize
from source.inputters.field import TextField, FixedField
from source.inputters.field import NumberField

from source.inputters.dataset import WithBowDataset


class SrcTgtCorpus(Corpus):
    """
    SrcTgtCorpus
    """

    def __init__(self, config):
        super(SrcTgtCorpus, self).__init__(config)
        self.min_len = config.min_len
        self.max_len = config.max_len
        self.embed_file = config.embed_file
        self.share_vocab = config.share_vocab

        self.SRC = TextField(tokenize_fn=tokenize,
                             embed_file=self.embed_file)
        if self.share_vocab:
            self.TGT = self.SRC
        else:
            self.TGT = TextField(tokenize_fn=tokenize,
                                 embed_file=self.embed_file)
        self.OUTPUT = self.TGT
        self.fields = {'src': self.SRC, 'tgt': self.TGT, 'output': self.OUTPUT}

        def src_filter_pred(src):
            return self.min_len <= len(self.SRC.tokenize_fn(src)) <= self.max_len

        def tgt_filter_pred(tgt):
            return self.min_len <= len(self.TGT.tokenize_fn(tgt)) <= self.max_len

        # self.filter_pred = lambda ex: src_filter_pred(ex['src']) and tgt_filter_pred(ex['tgt'])
        self.filter_pred = None

    def read_data(self, data_file, data_type="train"):
        data = []
        filtered = 0

        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                src = json.loads(line.strip())['src']
                tgt = json.loads(line.strip())['tgt']
                data.append({'src': src, 'tgt': tgt})

        filtered_num = len(data)
        if self.filter_pred is not None:
            data = [ex for ex in data if self.filter_pred(ex)]
        filtered_num -= len(data)
        print(
            "Read {} {} examples ({} filtered)".format(len(data), data_type.upper(), filtered_num))
        return data




class TopicGuide2Corpus(Corpus):
    """
    CueCorpus
    """

    def __init__(self, config):
        super(TopicGuide2Corpus, self).__init__(config)
        self.min_len = config.min_len
        self.max_len = config.max_len
        self.share_vocab = config.share_vocab
        self.embed_file = config.embed_file
        self.topic_words_num = config.topic_words_num
        self.topic_vocab_file = config.topic_vocab_file

        self.SRC = TextField(tokenize_fn=tokenize, embed_file=self.embed_file)

        self.TGT = TextField(tokenize_fn=tokenize, embed_file=self.embed_file)

        self.LABEL = NumberField(dtype=int)
        self.OUTPUT = TextField(tokenize_fn=tokenize, embed_file=self.embed_file)

        self.TOPIC = FixedField(fix_file=self.topic_vocab_file, embed_file=self.embed_file)

        self.fields = {
            'src': self.SRC,
            'tgt': self.TGT,
            'output': self.OUTPUT,
            'topic_src_label': self.LABEL,
            'topic_tgt_label': self.LABEL,
            'topic': self.TOPIC,
        }

        def src_filter_pred(src):
            return min_len <= len(self.SRC.tokenize_fn(src)) <= max_len

        def tgt_filter_pred(tgt):
            return min_len <= len(self.TGT.tokenize_fn(tgt)) <= max_len

        # self.filter_pred = lambda ex: src_filter_pred(ex['src']) and tgt_filter_pred(ex['tgt'])
        self.filter_pred = None

    def read_data(self, data_file, data_type="train"):
        """
        read_data
        """
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line.strip())
                src = line['src']
                tgt = line['tgt']
                topic_src_label = line['topic_src_label']
                topic_tgt_label = line['topic_tgt_label']

                data.append({
                    'src': src,
                    'tgt': tgt,
                    'output': tgt,
                    'topic_src_label': topic_src_label,
                    'topic_tgt_label': topic_tgt_label,
                    'topic': src,
                })

        filtered_num = len(data)
        if self.filter_pred is not None:
            data = [ex for ex in data if self.filter_pred(ex)]
        filtered_num -= len(data)
        print(
            "Read {} {} examples ({} filtered)".format(len(data), data_type.upper(), filtered_num))
        return data

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
        field_dict = {}
        for name, field in self.fields.items():
            if name == 'topic':
                field.build_vocab()
                field_dict[name] = field
                continue

            if field in field_data_dict:
                print("Building vocabulary of field {} ...".format(name.upper()))
                if field.vocab_size == 0:
                    if name != 'output':
                        field.build_vocab(field_data_dict[field],
                                          min_freq=self.min_freq,
                                          max_size=self.max_vocab_size)
                    field_dict[name] = field

        field_dict['output'].add_with_other_field(field_dict['tgt'])
        field_dict['output'].add_with_other_field(field_dict['topic'])
        if self.embed_file is not None:
            field_dict['output'].embeddings = field_dict['output'].build_word_embeddings(self.embed_file)

        for name, field in field_dict.items():
            vocab_dict[name] = field.dump_vocab()

        return vocab_dict

    def load_vocab(self, prepared_vocab_file):
        super().load_vocab(prepared_vocab_file)

        self.topic_bow_vocab_size = self.TOPIC.vocab_size
        self.Dataset = lambda x: WithBowDataset(data=x, bow_vocab_size=self.topic_bow_vocab_size)
