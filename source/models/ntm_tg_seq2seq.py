#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
"""
File: source/models/ntm_tg_seq2seq.py
"""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
import numpy as np

from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder, HRNNEncoder
from source.modules.decoders.topic_guide_rnn_decoder3 import RNNDecoder as RNNDecoder_full
from source.modules.decoders.topic_guide_rnn_decoder2 import RNNDecoder as RNNDecoder_old

from source.utils.criterions import NLLLoss
from source.utils.metrics import accuracy
from source.utils.misc import Pack

from source.modules.ntm.NTM import NTMR
from source.modules.ntm.topic_utils import get_mlp, Topics, NormalParameter


class Seq2Seq(BaseModel):
    """
    CMTE-ETA mode
    """

    def __init__(self, corpus, config):
        super(Seq2Seq, self).__init__()

        self.src_vocab_size = corpus.SRC.vocab_size
        self.tgt_vocab_size = corpus.TGT.vocab_size
        self.output_vocab_size = corpus.OUTPUT.vocab_size
        self.topic_vocab_size = corpus.TOPIC.vocab_size
        self.padding_idx = corpus.padding_idx

        self.embed_size = config.embed_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.attn_mode = config.attn_mode
        self.attn_hidden_size = config.attn_hidden_size

        self.with_bridge = config.with_bridge
        self.tie_embedding = config.tie_embedding
        self.dropout = config.dropout
        self.use_gpu = config.use_gpu
        self.decoder_attention_channels = config.decoder_attention_channels

        self.topic_k = config.topic_k
        self.topic_num = config.topic_num
        self.without_fd = config.without_fd

        # topic
        self.build_neural_topic_model(corpus)

        # encoder
        enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size,
                                padding_idx=self.padding_idx)

        self.encoder = RNNEncoder(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  embedder=enc_embedder,
                                  num_layers=self.num_layers,
                                  bidirectional=self.bidirectional,
                                  dropout=self.dropout)
        # bridge
        if self.with_bridge:
            self.bridge = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )

        # decoder
        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            dec_embedder = enc_embedder
        else:
            dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                    embedding_dim=self.embed_size,
                                    padding_idx=self.padding_idx)

        topic_vocab_size_ = self.topic_vocab_size
        if config.without_topic_project:
            topic_vocab_size_ = None
        if config.without_fd is not None and config.without_fd is True:
            RNNDecoder = RNNDecoder_old
            topic_vocab_size_ = None
        else:
            RNNDecoder = RNNDecoder_full
        self.decoder = RNNDecoder(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  output_size=self.output_vocab_size,
                                  topic_size=topic_vocab_size_,
                                  trans_mat=self.trans_mat,
                                  embedder=dec_embedder,
                                  num_layers=self.num_layers,
                                  attn_mode=self.attn_mode,
                                  attn_hidden_size=self.attn_hidden_size,
                                  memory_size=self.hidden_size,
                                  feature_size=None,
                                  dropout=self.dropout,
                                  tgt_unk_idx=corpus.OUTPUT.itos.index('<unk>'),
                                  attention_channels=self.decoder_attention_channels, )

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

    def build_neural_topic_model(self, corpus):

        self.build_trans_mat(corpus.OUTPUT.itos, corpus.TOPIC.itos)
        self.topic_embedder = Embedder(num_embeddings=self.topic_vocab_size,
                                       embedding_dim=self.embed_size,
                                       padding_idx=self.padding_idx)
        self.topic_layer = nn.Sequential(
            self.topic_embedder,
            nn.Linear(self.embed_size, self.hidden_size),
            nn.Tanh(),
        )

        hidden = get_mlp([self.topic_vocab_size, 800, self.embed_size], 'Sigmoid')
        normal = NormalParameter(self.embed_size, 50)
        h_to_z = nn.Sequential(
            nn.Linear(50, self.topic_num),
            nn.Dropout(0.2)
        )
        topics = Topics(self.topic_num, self.topic_vocab_size)
        penalty = 0.8
        self.ntm = NTMR(
            hidden=hidden,
            normal=normal,
            h_to_z=h_to_z,
            topics=topics,
            embedding=self.topic_embedder,
            penalty=penalty
        )

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
        self.trans_mat = O

    # def update_topicK(self):
    #     pass

    def encode(self, inputs, hidden=None, is_training=False):
        """
        encode
        """
        '''
        inputs: src, topic_src, topic_tgt, [tgt]
        '''
        outputs = Pack()
        enc_inputs = _, lengths = inputs.src[0][:, 1:-1], inputs.src[1] - 2

        enc_outputs, enc_hidden = self.encoder(enc_inputs, hidden)
        if self.with_bridge:
            enc_hidden = self.bridge(enc_hidden)

        guide_score = enc_hidden[-1]
        decoder_init_state = enc_hidden

        bow_src = inputs.bow
        ntm_stat = self.ntm(bow_src)
        outputs.add(ntm_loss=ntm_stat['loss'])

        # obtain topic words     
        _, tw_indices = self.ntm.get_topics().topk(self.topic_k, dim=1)  # K * k

        src_labels = F.one_hot(inputs.topic_src_label, num_classes=self.topic_num)  # B * K
        tgt_labels = F.one_hot(inputs.topic_tgt_label, num_classes=self.topic_num)
        src_words = src_labels.float() @ tw_indices.float()  # B * k
        src_words = src_words.detach().long()
        tgt_words = tgt_labels.float() @ tw_indices.float()
        tgt_words = tgt_words.detach().long()

        # only src topic word
        src_outputs = self.topic_layer(src_words)  # b * k * h

        # only tgt topic word
        tgt_outputs = self.topic_layer(tgt_words)  # b * k * h

        dec_init_state = self.decoder.initialize_state(
            hidden=decoder_init_state,
            attn_memory=enc_outputs,
            memory_lengths=lengths,
            guide_score=guide_score,
            src_memory=src_outputs,
            tgt_memory=tgt_outputs, )
        return outputs, dec_init_state

    def decode(self, input, state):
        """
        decode
        """
        log_prob, state, output = self.decoder.decode(input, state)
        return log_prob, state, output

    def forward(self, enc_inputs, dec_inputs, hidden=None, is_training=True):
        """
        forward
        """
        outputs, dec_init_state = self.encode(enc_inputs, hidden, is_training=is_training)
        log_probs, _ = self.decoder(dec_inputs, dec_init_state)
        outputs.add(logits=log_probs)
        return outputs

    def collect_metrics(self, outputs, target, bridge=None, epoch=-1):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        # response generation
        logits = outputs.logits
        nll = self.nll_loss(logits, target)
        num_words = target.ne(self.padding_idx).sum().item()
        acc = accuracy(logits, target, padding_idx=self.padding_idx)
        metrics.add(nll=(nll, num_words), acc=acc)
        loss += nll

        # neural topic model
        ntm_loss = outputs.ntm_loss.sum().item()
        loss += ntm_loss / self.topic_vocab_size * 0.3

        metrics.add(loss=loss)
        return metrics

    def iterate(self, inputs, optimizer=None, grad_clip=None, is_training=True, epoch=-1):
        """
        iterate
        """
        enc_inputs = inputs
        dec_inputs = inputs.tgt[0][:, :-1], inputs.tgt[1] - 1
        target = inputs.output[0][:, 1:]

        outputs = self.forward(enc_inputs, dec_inputs, is_training=is_training)

        metrics = self.collect_metrics(outputs, target, epoch=epoch)

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
