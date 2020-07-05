from .base_dataset import BaseDataset
from utils.util import load_pickle_file
from torchvision import transforms
from nltk.tokenize import RegexpTokenizer
import random
import os
import numpy as np
from PIL import Image

class COCODataset(BaseDataset):

    def __init__(self, opt, is_for_train=True):

        super(COCODataset, self).__init__(opt, is_for_train)
        self._name = opt.option_name
        self._is_train = is_for_train
        self._image_floder = 'train2014' if is_for_train else 'val2014'
        self._imsizes = [64, 128, 256]
        self._load_captions()
        self._class_id = np.arange(len(self._keys))

    def _load_captions(self):

        if self._is_train:
            self._pseudos = load_pickle_file(
                os.path.join(self._root, 'coco/pseudo/pseudos.pkl'))
            self._keys = list(self._pseudos.keys())
            self._corpus = load_pickle_file(
                os.path.join(self._root, 'coco/format_corpus.pkl'))
            self._corpus_length = len(self._corpus)

        else:
            self._captions = load_pickle_file('val.pkl')
            self._keys = list(self._captions.keys())

    def _get_caption_info(self, key):
            
        if self._is_train:
            pseudo, pseudo_concepts = self._pseudos[key]

            self._sample['pseudos'], self._sample['pseudo_lens'] = \
                self._pad(pseudo, 20)
            self._sample['pseudo_concepts'], self._sample['pseudo_concept_lens'] = \
                self._pad(pseudo_concepts, 5)

            ix = random.randint(0, self._corpus_length - 10)
            corpus_concepts, corpus = self._corpus[ix]
            self._sample['corpus'], self._sample['corpus_len'] = \
                self._pad(corpus[1:-1], 20)

            self._sample['corpus_concepts'], self._sample['corpus_len'] =\
                self._pad(corpus_concepts, 5)

        else:
            caption, concepts = self._captions[key]

            self._sample['caption'], self._sample['caption_len'] = \
                self._pad(caption, 20)

    def _get_image_info(self, key):

        images = []
        image_path = os.path.join(self._root, 'coco/images/train2014/%s.jpg' % key)
        image = Image.open(image_path).convert('RGB')
        image = self._transform(image)
        for imsize in self._imsizes:
            if imsize < self._imsizes[-1]:
                ret_image = transforms.Resize(imsize)(image)
            else:
                ret_image = image
            ret_image = self._normalize(ret_image)
            images.append(ret_image)

        self._sample['images'] = images

    def __getitem__(self, index):

        self._sample = {}
        key = self._keys[index]
        self._get_image_info(key)
        self._get_caption_info(key)
        self._sample['key'] = key
        self._sample['class_id'] = self._class_id[index]

        return self._sample

    def __len__(self):
        return len(self._keys)
