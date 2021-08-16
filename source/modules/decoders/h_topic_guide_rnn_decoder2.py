#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
"""
File: source/decoders/hgfu_rnn_decoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from source.modules.attention import Attention
from source.modules.decoders.state import DecoderState
from source.utils.misc import Pack
from source.utils.misc import sequence_mask


class RNNDecoder(nn.Module):
    """
    A topic guide  GRU recurrent neural network decoder.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 topic_size,
                 output_size,
                 trans_mat,
                 embedder=None,
                 num_layers=1,
                 attn_mode=None,
                 attn_hidden_size=None,
                 memory_size=None,
                 feature_size=None,
                 dropout=0.0,
                 tgt_unk_idx=3,
                 attention_channels='ST',
                 ):
        super(RNNDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.topic_size = topic_size
        self.output_size = output_size
        self.trans_mat = trans_mat.transpose(0, 1)
        self.embedder = embedder
        self.num_layers = num_layers
        self.attn_mode = None if attn_mode == 'none' else attn_mode
        self.attn_hidden_size = attn_hidden_size or hidden_size // 2
        self.memory_size = memory_size or hidden_size
        self.feature_size = feature_size
        self.dropout = dropout
        self.tgt_unk_idx = tgt_unk_idx
        self.attention_channels = attention_channels

        self.rnn_input_size = self.input_size + self.memory_size * 2

        self.out_input_size = self.memory_size + self.hidden_size
        self.topic_input_size = self.hidden_size + self.memory_size
        
        self.soft_prob_layer = nn.Linear(self.hidden_size, 2)

        self.tgv_layer = nn.Sequential(
            nn.Linear(self.memory_size * 2, self.memory_size),
            nn.Tanh()
        )
        self.tgv_fc = nn.Linear(self.memory_size*2, self.memory_size)

        self.attention = Attention(query_size=self.hidden_size,
                                   memory_size=self.memory_size,
                                   hidden_size=self.attn_hidden_size,
                                   mode=self.attn_mode,
                                   project=False)

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)

        attention_channels_length = len(self.attention_channels)
        self.tgv_layer = nn.Sequential(
            nn.Linear(self.memory_size * attention_channels_length, self.memory_size),
            nn.Tanh()
        )

        self.output_out_layer = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.out_input_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.output_size),
        )

        if self.topic_size is not None:
            self.topic_out_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.topic_input_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.topic_size),
            )
        self.lsf = nn.LogSoftmax(dim=-1)

    def output_layer(self, out_input, topic_input, state_vectors):
        out_logits = self.output_out_layer(out_input)

        if self.topic_size is not None:
            topic_logits = self.topic_out_layer(topic_input)
            topic_logits = F.linear(topic_logits, self.trans_mat)
            soft_probs = self.soft_prob_layer(state_vectors)
            out_logit_weight = soft_probs[:, :, 0].unsqueeze(-1).repeat(1, 1, self.output_size)
            topic_logit_weight = soft_probs[:, :, 1].unsqueeze(-1).repeat(1, 1, self.output_size)
            logits = out_logit_weight * out_logits + topic_logit_weight * topic_logits
        else:
            logits = out_logits

        logits = self.lsf(logits)
        return logits

    def initialize_state(self,
                         hidden,
                         attn_memory=None,
                         attn_mask=None,
                         memory_lengths=None,
                         guide_score=None,
                         topic_feature=None, ):
        """
        initialize_state
        """
        if self.attn_mode is not None:
            assert attn_memory is not None

        if memory_lengths is not None and attn_mask is None:
            max_len = attn_memory.size(1)
            attn_mask = sequence_mask(memory_lengths, max_len).eq(0)

        init_state = DecoderState(
            hidden=hidden,
            attn_memory=attn_memory,
            attn_mask=attn_mask,
            topic_feature=topic_feature.unsqueeze(1),
            # bridge_memory=bridge_memory,
            guide_score=guide_score,
            state_vector=None,
        )
        return init_state

    def decode(self, input, state, is_training=False):
        """
        decode
        """
        hidden = state.hidden
        rnn_input_list = []
        topic_rnn_input_list = []
        topic_input_list = []
        out_input_list = []
        output = Pack()
        if not is_training:
            tgt_vocab_size = self.embedder.num_embeddings
            input[input >= tgt_vocab_size] = self.tgt_unk_idx
        # print(torch.max(input))

        if self.embedder is not None:
            input = self.embedder(input)

        # shape: (batch_size, 1, input_size)
        input = input.unsqueeze(1)
        rnn_input_list.append(input)

        # context vector
        attn_memory = state.attn_memory
        attn_mask = state.attn_mask

        query = hidden[-1].unsqueeze(1)
        weighted_context, attn = self.attention(query=query, memory=attn_memory, mask=attn_mask)
        rnn_input_list.append(weighted_context)
        topic_rnn_input_list.append(weighted_context)

        # guide vector 
        tgv_inputs = torch.cat([state.topic_feature, query], dim=-1)
        topic_guide_vector = self.tgv_layer(tgv_inputs)
        rnn_input_list.append(topic_guide_vector)

        # output
        rnn_input = torch.cat(rnn_input_list, dim=-1)
        rnn_output, new_hidden = self.rnn(rnn_input, hidden)

        out_input_list.append(weighted_context)
        out_input_list.append(rnn_output)

        topic_input_list.append(weighted_context)
        topic_input_list.append(rnn_output)

        out_input = torch.cat(out_input_list, dim=-1)
        topic_input = torch.cat(topic_input_list, dim=-1)
        state.hidden = new_hidden
        state_vector = torch.cat([new_hidden.transpose(0, 1), topic_guide_vector], dim=-1)
        state.state_vector = self.tgv_fc(state_vector)

        
        if is_training:
            return out_input, topic_input, state, output
        else:
            log_prob = self.output_layer(out_input, topic_input, state.state_vector)
            return log_prob, state, output

    def forward(self, inputs, state):
        """
        forward
        """
        inputs, lengths = inputs
        batch_size, max_len = inputs.size()

        out_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.out_input_size),
            dtype=torch.float)
        topic_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.topic_input_size),
            dtype=torch.float)
        state_vectors = inputs.new_zeros(
            size=(batch_size, max_len, self.hidden_size ),
            dtype=torch.float)

        # sort by lengths
        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)
        state = state.index_select(indices)

        # number of valid input (i.e. not padding index) in each time step
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)

        for i, num_valid in enumerate(num_valid_list):
            dec_input = inputs[:num_valid, i]
            valid_state = state.slice_select(num_valid)
            out_input, topic_input, valid_state, _ = self.decode(
                dec_input, valid_state, is_training=True)
            state.hidden[:, :num_valid] = valid_state.hidden
            out_inputs[:num_valid, i] = out_input.squeeze(1)
            topic_inputs[:num_valid, i] = topic_input.squeeze(1)
            state_vectors[:num_valid, i] = valid_state.state_vector.squeeze(1)

        # Resort
        _, inv_indices = indices.sort()
        state = state.index_select(inv_indices)
        out_inputs = out_inputs.index_select(0, inv_indices)
        topic_inputs = topic_inputs.index_select(0, inv_indices)
        state_vectors = state_vectors.index_select(0, inv_indices)

        log_probs = self.output_layer(out_inputs, topic_inputs, state_vectors)
        return log_probs, state
