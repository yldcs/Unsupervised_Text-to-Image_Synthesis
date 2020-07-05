#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .base_dataset import BaseDataset
from utils.util import load_pickle_file
import os

class Con2SenTrainDataset(BaseDataset):
    def __init__(self, opt, is_for_train=True):
        super(Con2SenTrainDataset, self).__init__(opt, is_for_train)
        self._opt = opt
        self._data_dir = self._opt.data_dir
        self._load_data()

    def _load_data(self):
        file_path = os.path.join(self._data_dir, 'coco/format_corpus.pkl')
        self._corpus = load_pickle_file(file_path)

    def __getitem__(self, ix):
        concepts, captions = self._corpus[ix]
        captions, caption_lens = self._pad(captions, 20)
        concepts, concept_lens = self._pad(concepts, 5)

        sample = {'captions': captions, 'caption_lens' : caption_lens,
                  'concepts': concepts, 'concept_lens': concept_lens}
        return sample

    def __len__(self):
        return len(self._corpus)

