#!/usr/bin/env python
# -*- coding: utf-8 -*-

from options import OptionFactory
from datasets import CustomDatasetDataLoader
from models import ModelsFactory
import time
import sys

class Train(object):

    def __init__(self, option_name):

        self._opt = OptionFactory.get_by_name(option_name).parse()
        self._model = ModelsFactory.get_by_name(self._opt.model_name, self._opt)
        self._dataloader_train = CustomDatasetDataLoader(self._opt, is_for_train=True).get_dataloader()
        self._it = 0
        self._total_batches = len(self._dataloader_train)
        self._train()

    def _train(self):

        for i_epoch in range(self._opt.load_epoch + 1, self._opt.total_epoch):

            if self._it > self._opt.total_it:
                break

            epoch_start_time = time.time()

            # update learning rate
            self._model.update_learning_rate(i_epoch)

            # train epoch
            self._train_epoch(i_epoch)

            # save model
            print('saving the model at the end of epoch %d, iters %d' % 
                  (i_epoch, self._opt.total_epoch))
            if i_epoch + 1 % self._opt.epoch_interval == 0:
                self._model.save('epoch_%d' % i_epoch)
            
            # print epoch info
            epoch_time = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self._opt.total_epoch, epoch_time, epoch_time/60, epoch_time/3600))

    def _train_epoch(self, i_epoch):

        self._model.set_train()

        for i_train_batch, train_batch in enumerate(self._dataloader_train):
            batch_start_time = time.time()
            self._model.set_input(train_batch)

            self._model.optimize_parameters()

            self._it += 1

            elasped = time.time() - batch_start_time
            if self._it % self._opt.display_interval == 0:
                self._model.display_terminal(i_epoch, self._opt.total_epoch,
                         i_train_batch, self._total_batches, elasped)

            if self._it + 1 % self._opt.it_interval == 0:
                self._model.save('it_%d' % self._it)

         
        
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    Train(sys.argv[2])
