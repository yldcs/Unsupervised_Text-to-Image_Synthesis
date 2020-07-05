#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from torch.utils import data 
import torch.nn as nn
from collections import OrderedDict
import sys
if sys.version_info[0]:
    import cPickle as pkl
else:
    import _pickle as pkl

class CorpusDataset(data.Dataset):
    def __init__(self, vocab):
        super(CorpusDataset, self).__init__()
        
        self.vocab = vocab 
        with open('corpus.pkl', 'rb') as f:
            corpus = pkl.load(f)
        self.objs_dict = self.load_obj()
        self.objs = self.objs_dict.keys()
        self.corpus = self.preprocess(corpus)
        self.num = len(self.corpus)
        self.class_ids = np.arange(self.num)
    def load_obj(self):
        with open('../data/coco/objs.names', 'r') as f:
            objs = [i.strip().lower().split(' ')[-1] for i in list(f)] 
        objs_dict = OrderedDict() 
        it = 0
        for ob in objs:
            if ob not in objs_dict:
                objs_dict[ob] = it 
                it += 1
        return objs_dict
    def preprocess(self, corpus):
        data = []
        for cap in corpus: 
            cons = self.get_cons(cap)
            #  cons = con_preprocess(cons)
            if len(cons) > 0:
                if len(cap) > 18:
                    cap = cap[:18]
                data.append([cons, cap])
        return data
    def get_cons(self, sent):
        sent = set(sent) 
        cons = sent.intersection(self.objs)
        return cons
    def pad(self, words, ln):
        out = np.zeros((ln), dtype=np.int64)
        num = len(words)
        if num > ln:
            num = ln
        out[:num] = words[:num]
        return out, num
    def __getitem__(self, ix):
        cons, sentence = self.corpus[ix]
        cons = [self.objs_dict[i] for i in cons]
        sentence = [self.vocab.word_to_id(i) for i in sentence]
        sentence, sent_len = self.pad(sentence, 20)
        cons, con_len = self.pad(cons, 5)
        cls = self.class_ids[ix]
        return  sentence, sent_len, cons, con_len, cls
    def __len__(self):
        return len(self.corpus)



