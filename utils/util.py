#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import _pickle as pickle
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)




def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def cleardir(path):
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)

def load_pickle_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data

def write_pickle_file(pkl_path, data_dict):
    dirname = os.path.dirname(pkl_path)
    if not os.path.exists(dirname):
        mkdir(dirname)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data_dict, f, protocol=2)


def idxs2sentences(idxs, vocab):
    '''
    Input :
        idsx : (B, 20)
        ix2word : (vacob_size)
    '''
    sentences = []
    bos = vocab.word_to_id('<S>')
    eos = vocab.word_to_id('</S>')
    for idx in idxs:
        sentence = []
        if not isinstance(idx, list):
            idx = [idx]
        for ix in idx:
            if ix in [eos, 0]:
               break
            if ix == bos:
               continue
            sentence.append(vocab.id_to_word(ix))
        if len(sentence) > 0:
            sentences.append(" ".join(sentence))
        else:
            sentences.append("<UNK>")

    return sentences
