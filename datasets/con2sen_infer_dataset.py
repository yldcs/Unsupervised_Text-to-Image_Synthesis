#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base_dataset import BaseDataset
from utils.util import load_pickle_file
import os

class Con2SenInferDataset(BaseDataset):
    
    def __init__(self, opt, is_for_train):

        super(Con2SenInferDataset, self).__init__(opt, is_for_train)
        self._detect = load_pickle_file(
            os.path.join(self._root, 'coco/format_coco_concepts.pkl'))

        self._keys = list(self._detect.keys())

    def __getitem__(self, index):
        key = self._keys[index]
        concepts = self._detect[key]
        concepts, concept_lens = self._pad(concepts, 5)
        sample = {'key' : key, 'concepts' : concepts, 'concept_lens': concept_lens}
        return sample

    def __len__(self):
        return len(self._keys)
