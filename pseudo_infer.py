#!/usr/bin/env python
# -*- coding: utf-8 -*-

from options import OptionFactory
from utils.util import write_pickle_file
from datasets import CustomDatasetDataLoader
from models import ModelsFactory
import time
import sys

class PseudoInfer(object):

    def __init__(self, option_name):

        self._opt = OptionFactory.get_by_name(option_name).parse()
        self._model = ModelsFactory.get_by_name(self._opt.model_name, self._opt)
        self._dataloader_infer = CustomDatasetDataLoader(self._opt, is_for_train=False).get_dataloader()
        self._total_batches = len(self._dataloader_infer)
        self._infer()

    def _infer(self):

        self._model.load()
        self._model.set_eval()
        self._last_time = time.time()
        for i_train_batch, train_batch in enumerate(self._dataloader_infer):

            self._model.set_input(train_batch)

            self._model.forward()

            elasped = time.time() - self._last_time
            self._model.display_terminal(i_train_batch, self._total_batches, elasped)
            
        write_pickle_file(self._opt.save_path, self._model.info)

if __name__ == '__main__':
    PseudoInfer(sys.argv[2])
