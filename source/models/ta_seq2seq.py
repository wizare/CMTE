#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
"""
File: source/models/ta_seq2seq.py
"""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder, HRNNEncoder
from source.modules.decoders.topic_aware_rnn_decoder import RNNDecoder
from source.utils.criterions import NLLLoss
from source.utils.metrics import accuracy
from source.utils.misc import Pack


class Seq2Seq(BaseModel):
    """
    Seq2Seq
    """

    def __init__(self, corpus, config):
        super(Seq2Seq, self).__init__()

        self.src_vocab_size = corpus.SRC.vocab_size
        self.tgt_vocab_size = corpus.TGT.vocab_size
        self.output_vocab_size = corpus.OUTPUT.vocab_size
        self.topic_vocab_size = corpus.TOPIC.vocab_size

        self.embed_size = config.embed_size
        self.hidden_size = config.hidden_size
        self.padding_idx = corpus.padding_idx
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.attn_mode = config.attn_mode
        self.attn_hidden_size = config.attn_hidden_size
        self.with_bridge = config.with_bridge
        self.tie_embedding = config.tie_embedding
        self.dropout = config.dropout
        self.use_gpu = config.use_gpu

        self.trans_mat = self.build_trans_mat(corpus.OUTPUT.itos, corpus.TOPIC.itos)
        self.topic_embedder = Embedder(num_embeddings=self.topic_vocab_size,
                                       embedding_dim=self.embed_size,
                                       padding_idx=self.padding_idx)
        self.topic_bridge = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.Tanh(),
        )

        enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size,
                                padding_idx=self.padding_idx)

        self.encoder = RNNEncoder(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  embedder=enc_embedder,
                                  num_layers=self.num_layers,
                                  bidirectional=self.bidirectional,
                                  dropout=self.dropout)

        if self.with_bridge:
            self.bridge = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            dec_embedder = enc_embedder
        else:
            dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                    embedding_dim=self.embed_size,
                                    padding_idx=self.padding_idx)

        self.decoder = RNNDecoder(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  output_size=self.output_vocab_size,
                                  topic_size=self.topic_vocab_size,
                                  trans_mat=self.trans_mat,
                                  embedder=dec_embedder,
                                  num_layers=self.num_layers,
                                  attn_mode=self.attn_mode,
                                  attn_hidden_size=self.attn_hidden_size,
                                  memory_size=self.hidden_size,
                                  feature_size=None,
                                  dropout=self.dropout,
                                  tgt_unk_idx=corpus.OUTPUT.itos.index('<unk>'))

        # Loss Definition
        if self.padding_idx is not None:
            weight = torch.ones(self.output_vocab_size)
            weight[self.padding_idx] = 0
        else:
            weight = None
        self.nll_loss = NLLLoss(weight=weight,
                                ignore_index=self.padding_idx,
                                reduction='mean')

        if self.use_gpu:
            self.cuda()

    def build_trans_mat(self, output_itos, topic_itos, ):
        O = torch.eye(self.output_vocab_size)
        tIndex = []
        unk_index = output_itos.index('<unk>')
        for ts in topic_itos:
            try:
                ti = output_itos.index(ts)
            except Exception as e:
                ti = unk_index
            tIndex.append(ti)
        O = O[tIndex]
        if self.use_gpu:
            O = O.cuda()
        return O

    def encode(self, inputs, hidden=None):
        """
        encode
        """
        outputs = Pack()
        enc_inputs = _, lengths = inputs.src[0][:, 1:-1], inputs.src[1] - 2
        enc_outputs, enc_hidden = self.encoder(enc_inputs, hidden)

        if self.with_bridge:
            enc_hidden = self.bridge(enc_hidden)

        # topic word
        topic_inputs, topic_lengths = inputs.topic
        topic_lengths -= 2
        topic_outputs = self.topic_embedder(topic_inputs[:, 1:-1])  # b * len * h
        topic_outputs = self.topic_bridge(topic_outputs)

        dec_init_state = self.decoder.initialize_state(
            hidden=enc_hidden,
            attn_memory=enc_outputs,
            memory_lengths=lengths,
            topic_memory=topic_outputs,
            topic_memory_lengths=topic_lengths)
        return outputs, dec_init_state

    def decode(self, input, state):
        """
        decode
        """
        log_prob, state, output = self.decoder.decode(input, state)
        return log_prob, state, output

    def forward(self, enc_inputs, dec_inputs, hidden=None):
        """
        forward
        """
        outputs, dec_init_state = self.encode(enc_inputs, hidden)
        log_probs, _ = self.decoder(dec_inputs, dec_init_state)
        outputs.add(logits=log_probs)
        return outputs

    def collect_metrics(self, outputs, target):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        logits = outputs.logits
        nll = self.nll_loss(logits, target)
        num_words = target.ne(self.padding_idx).sum().item()
        acc = accuracy(logits, target, padding_idx=self.padding_idx)
        metrics.add(nll=(nll, num_words), acc=acc)
        loss += nll

        metrics.add(loss=loss)
        return metrics

    def iterate(self, inputs, optimizer=None, grad_clip=None, is_training=True, epoch=-1):
        """
        iterate
        """
        enc_inputs = inputs
        dec_inputs = inputs.tgt[0][:, :-1], inputs.tgt[1] - 1
        target = inputs.output[0][:, 1:]

        outputs = self.forward(enc_inputs, dec_inputs)
        metrics = self.collect_metrics(outputs, target)

        loss = metrics.loss
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        if is_training:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(parameters=self.parameters(),
                                max_norm=grad_clip)
            optimizer.step()
        return metrics, None
