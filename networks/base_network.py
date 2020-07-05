#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BaseNetwork(nn.Module):

    def __init__(self):
        super(BaseNetwork, self).__init__()
        self._name = 'BaseNetwork'
    
    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.data.fill_(0.0)

        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    def forward(self, *input):
        raise NotImplementedError


