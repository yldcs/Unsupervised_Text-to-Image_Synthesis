#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

class CustomDatasetDataLoader(object):

    def __init__(self, opt, is_for_train=True):

        self._opt = opt
        self._is_for_train = is_for_train
        self._num_workers = opt.num_workers_train if is_for_train else opt.num_workers_test
        self._shuffle = is_for_train
        self._create_dataset_dataloader()

    def _create_dataset_dataloader(self):

        self._dataset = DatasetFactory.get_by_name(self._opt.dataset_name, self._opt, self._is_for_train)
        self._dataloader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self._opt.batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            drop_last=self._opt.drop_last)

    def get_dataset(self):
        return self._dataset

    def get_dataloader(self):
        return self._dataloader
    
class DatasetFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, *args, **kwargs):
        if dataset_name == 'base':
            from .base_dataset import BaseDataset
            dataset = BaseDataset(*args, **kwargs)

        elif dataset_name == 'con2sen_train':
            from .con2sen_dataset import Con2SenTrainDataset
            dataset = Con2SenTrainDataset(*args, **kwargs)

        elif dataset_name == 'con2sen_infer':
            from .con2sen_infer_dataset import Con2SenInferDataset
            dataset = Con2SenInferDataset(*args, **kwargs)

        elif dataset_name in ['DAMSM', 'VCD', 'GSC', 'evaluation']:
            from .datasets import COCODataset
            dataset = COCODataset(*args, **kwargs)

        else:
            raise ValueError('Dataset [%s] not recognized.' % dataset_name)
        
        print('dataset [%s] was created, length: %d' % (dataset_name, len(dataset)))

        return dataset
