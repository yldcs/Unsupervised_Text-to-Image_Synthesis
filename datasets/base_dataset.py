#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torchvision.transforms as transforms
import torch.utils.data as data
from utils.vocabulary import Vocabulary
import os
import numpy as np

class BaseDataset(data.Dataset):
    
    def __init__(self, opt, is_for_train):
        super(BaseDataset, self).__init__()
        self._name = 'BaseDataset'
        self._root = opt.data_dir
        self._opt = opt
        self._is_for_train = is_for_train
        self._transform_normalize()
        self._vocab = Vocabulary( os.path.join(opt.data_dir, 'assets', opt.word_count))

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root

    def _transform_normalize(self, imsize=256, RandomCrop=True, RandomHorizontalFlip=True):
        
        sequence = []
        if RandomCrop:
            sequence += [transforms.Resize(int(imsize * 76 / 64)), transforms.RandomCrop(imsize)]
        else:
            sequence += [transforms.Resize(imsize)]
        if RandomHorizontalFlip:
            sequence += [transforms.RandomHorizontalFlip()]
        self._transform = transforms.Compose(sequence)

        sequence = [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        self._normalize = transforms.Compose(sequence)
        
    def get_transform(self):
        return self._transform

    def _pad(self, seq, ln=20):
        if not isinstance(seq, list):
            seq = seq.split(' ')
        seq = [self._vocab.word_to_id(i) for i in seq]
        pad_seq = np.zeros((20), dtype=np.int64)
        seq_len = len(seq)
        if seq_len > ln:
            seq_len = ln
        pad_seq[:seq_len] = seq[:seq_len]
        return pad_seq, seq_len

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def _read_dataset_paths(self):
        raise NotImplementedError

    def load_pair(self):
        raise NotImplementedError
