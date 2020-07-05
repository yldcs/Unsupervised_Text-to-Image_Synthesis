#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.embs = nn.Embedding(params.vocab_size, params.hidden_size) 
        self.rnn = nn.LSTM(params.input_size, params.hidden_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embs.weight.data.uniform_(-initrange, initrange)

    def forward(self, captions, cap_lens):
        captions = self.embs(captions)        
        captions = pack_padded_sequence(captions, lengths=cap_lens, enforce_sorted=False)        
        _, (hidden, cell) = self.rnn(captions)
        return hidden

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.embs = nn.Embedding(params.vocab_size, params.hidden_size) 
        self.rnn = nn.LSTM(params.input_size, params.hidden_size)
        self.generator = nn.Linear(params.hidden_size, params.vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embs.weight.data.uniform_(-initrange, initrange)

    def one_step(self, rnn_inp, hidden, cell):
        rnn_inp = self.embs(rnn_inp)
        _, (hidden, cell) = self.rnn(rnn_inp,(hidden,cell))
        log_p = F.log_softmax(self.generator(hidden.squeeze(0)))
        return log_p, (hidden,cell) 

    def forward(self, t, hidden, cell):
        t_embs = self.embs(t)
        rnn_out, state = self.rnn(t_embs, (hidden, cell))
        rnn_out = self.generator(rnn_out)
        return rnn_out
