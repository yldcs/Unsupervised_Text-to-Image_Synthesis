#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.util import load_pickle_file
import torch

class ob(object):
    def __init__(self, a):
        self.it = a
class Train(object):

    def __init__(self):

        b = ob(2)
        c = ob(3)
        self._a = [b, c]

    def forward(self):
        for p in self._a:
            print(p.it)
        self._set(self._a)
        for p in self._a:
            print(p.it)

    def _set(self, data):
        for p in data:
            p.it = 9

if __name__=='__main__':


    a = torch.tensor(10).float()
    b = torch.tensor(2).float()
    c = a.add_(b, 0.1)
    d = a.add_(0.1, b)
    print(c)
    print(d)
