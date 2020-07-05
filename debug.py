#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.util import load_pickle_file

image_concepts = load_pickle_file('./data/coco/pseudo/pseudos.pkl')

filenames = load_pickle_file('./data/coco/filenames/train_filenames.pickle')

pseudo = {}
for key in filenames[:50000]:
    print(key in image_concepts)
    if key in image_concepts:
        pseudo[key] = image_concepts[key]

print(len(list(pseudo.keys())))
