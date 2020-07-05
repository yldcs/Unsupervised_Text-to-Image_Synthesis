from .base_dataset import BaseDataset
from collections import OrderedDict 

import os
import h5py
import numpy as np
from PIL import Image
import numpy.random as random
from vocabulary import Vocabulary

def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):

    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    if transform is not None:
        img = transform(img)
    ret = []
    for i in range(3):
        # print(imsize[i])
        if i < (3 - 1):
            re_img = transforms.Resize(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))
    return ret

class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', base_size=64, 
                 transform=None, target_transform=None):

        #  self.data_dir = params.data_dir
        self.split = split
        #  self.base_size = params.base_size

        image_size = 256
        self.transform = transforms.Compose(
            [transforms.Scale(int(74 / 64 * image_size)),
             transforms.RandomCrop(size = image_size),
             transforms.RandomHorizontalFlip()])
    
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.vocab = Vocabulary('../data/coco_word_counts.txt') 
        self.target_transform = target_transform

        self.imsize = [64, 128, 256]
        self.data = []
        self.data_dir = data_dir
        self.pseudo = self.load_pseudo() 
        keys = self.load_filename() 
        self.objs_dict, self.objs = self.load_obj()
        self.detect = self.load_detect()
        keys = [i for i in keys if i in self.pseudo]
        self.keys = [i for i in keys if i in self.detect]
        self.objs_set = self.objs_dict.keys()
        self.num = len(self.keys)
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
        return objs_dict, objs
    def load_pseudo(self):
        with open('../data/3w_it_30000.pkl', 'rb') as f:
            train_coco = pkl.load(f)
        return train_coco  
    def load_filename(self):
        with open('../data/coco/train/filenames.pickle', 'rb') as f:
            filename = pkl.load(f)
        return filename[:50000]
    def pad_caption(self, cap):
        cap = [self.vocab.word_to_id(i) for i in cap]
        pad_cap = np.zeros((18), dtype = np.int64) 
        ln = len(cap)
        if ln > 18:
            ln = 18 
        pad_cap[:ln] = cap[:ln]
        return pad_cap, ln

    def get_text_con(self, cap):
        cap = set(cap) 
        con = cap.intersection(self.objs_set)
        con_cls = []
        for cn in con:
            con_cls.append(self.objs[cn])
        cls = np.zeros((5), dtype=np.int64)
        ln = len(con_cls)
        if ln > 5:
            ln = 5
        cls[:ln] = con_cls[:ln]
        return cls, ln
    def load_detect(self):
        detect = OrderedDict()
        with h5py.File('../data/object.hdf5') as f:
            for key, val in f.items():
                classes = val['detection_classes'][()]
                if len(classes) > 0:
                    cons = [self.objs[int(i) - 1] for i in classes] 
                    detect[key] = cons
        return detect
    def get_img_con(self, key):
        con_cls = []
        for cn in self.detect[key]:
            cls = self.objs_dict[cn]
            if cls not in con_cls:
                con_cls.append(cls)
        cls = np.zeros((5), dtype=np.int64)
        ln = len(con_cls)
        if ln > 5:
            ln = 5
        cls[:ln] = con_cls[:ln]
        return cls, ln
    def __getitem__(self, ix):
        filename = self.keys[ix]
        cls = self.class_ids[ix]
        image = get_imgs("../data/coco/image/train2014/" + filename +'.jpg', 
                         self.imsize, None, self.transform, self.norm) 
        pseudo_caption = self.pseudo[filename]
        if isinstance(pseudo_caption, str):
            pseudo_caption = pseudo_caption.split(' ') 
        con_cls, con_ln = self.get_img_con(filename)
        pseudo_caption, ln = self.pad_caption(pseudo_caption) 
        return image, pseudo_caption, ln, con_cls, con_ln, cls, filename 

    def __len__(self):
        return len(self.keys)

