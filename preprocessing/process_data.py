#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.util import load_pickle_file, write_pickle_file
from options import OptionFactory
import h5py
import sys
import os

class ProcessData(object):

    def __init__(self, option_name):

        self._opt = OptionFactory.get_by_name(option_name).parse()
        self._root = self._opt.data_dir
        self._save_dir = self._opt.outputs_dir
        self._load_concept()
        self._load_corpus()
        self._load_detect()

    def _load_corpus(self):

        filepath = os.path.join(self._root, 'coco/corpus.pkl')
        self._corpus = load_pickle_file(filepath)
        print(len(self._corpus))


    def _load_detect(self):

        train_filenames = load_pickle_file('./data/coco/filenames/train_filenames.pickle')
        train_filenames = train_filenames[:50000]
        self._detect = {}
        detect_path = os.path.join(self._root, 'coco/concepts.hdf5')
        with h5py.File(detect_path, 'r') as f:
            for key, val in f.items():
                classes = val['detection_classes'][()]
                if len(classes) > 0 and key in train_filenames:
                    concepts = [self._concepts_list[int(i) - 1] for i in classes]
                    self._detect[key] = concepts
        print(len(list(self._detect.keys())))
    def _get_cons(self, sent):

        sent = set(sent)
        cons = sent.intersection(self._concepts_set)
        return list(cons)

    def _load_concept(self):

        filepath = os.path.join(self._root, 'coco/concepts.names')
        with open(filepath, 'r') as f:
            self._concepts_list = [i.strip().lower().split(' ')[-1] for i in list(f)]

        self._concepts_set = set(self._concepts_list)

        self._concepts_dict = {}

        for i_concept, concept in enumerate(self._concepts_list):
            if concept not in self._concepts_dict:
                self._concepts_dict[concept] = i_concept

    def format_corpus(self):

        data = []
        for cap in self._corpus:
            if len(cap) > 18:
                cap = cap[:18]
            cap = ['<S>'] + cap + ['</S>']
            cons = self._get_cons(cap)
            if len(cons) > 0:
                data.append([cons, cap])
        filepath = os.path.join(self._save_dir, 'format_corpus.pkl')
        write_pickle_file(filepath, data)

    def format_coco_concepts(self):
        
        filepath = os.path.join(self._save_dir, 'format_coco_concepts.pkl')
        write_pickle_file(filepath, self._detect)

if __name__ == '__main__': 
    
    process = ProcessData(sys.argv[2])
    #  process.format_corpus()
    process.format_coco_concepts()
    #  detect = load_pickle_file('./data/coco/format_coco_concepts.pkl')
    #  print(detect)

