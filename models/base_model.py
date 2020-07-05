#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os

from utils.vocabulary import Vocabulary
from utils.util import mkdir

class BaseModel(object):
    
    def __init__(self, opt):

        self._name = 'BaseModel'
        self._gpu_ids = opt.gpu_ids
        self._Tensor = torch.cuda.FloatTensor if self._gpu_ids else torch.FloatTensor
        self.is_train = opt.is_train
        self._save_dir = opt.checkpoints_dir
        self._vocab = Vocabulary(os.path.join(opt.data_dir, 'assets', opt.word_count))

    @property
    def name(self):
        return self._name

    def set_input(self, input):
        assert False, 'set input not implemented'

    def _build_models(self):
        assert False, 'build_models not implemented'

    def _init_train_vars(self):
        assert False, '_init_train_vars not implemented'

    def _init_loss(self):
        assert False, '_init_loass not implemented'

    def set_train(self):
        assert False, 'set_train not implemented'
    
    def set_eval(self):
        assert False, 'set_eval not implemented'

    def forward(self, *input):
        assert False, 'forward not implemented'

    def optimize_parameters(self):
        assert False, 'optimize_parameters not implemented'

    def save(self):
        assert False, 'save not implemented'

    def load(self):
        assert False, 'laod not implemented'
    
    def _save_network(self, network, network_label, epoch_label):
        if not os.path.exists(self._save_dir):
            mkdir(self._save_dir)
        save_filename = 'net_%s_%s.pth' % (network_label, epoch_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        print('saved net: %s' % save_path)

    def _save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = 'opt_%s_%s.pth' % (optimizer_label, epoch_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)
        print('saved optimizer: %s' % save_path)

    def _load_network(self, network, network_label, epoch_label):
        load_filename = 'net_%s_%s.pth' % (network_label, epoch_label)
        load_path = os.path.join(self._save_dir, load_filename)
        self._load_params(network, load_path)
        print('load network from  %s' % load_path)

    def _load_params(self, network, load_path):

        assert os.path.exists(load_path), \
            'Weights file not found. Have you trained a model!? We can not find one %s' % load_path

        save_data = torch.load(load_path)

        network.load_state_dict(save_data)

        print("Loading net: %s" % load_path)


    def _load_optimizer(self, optimizer, network_label, epoch_label):
        load_filename = 'opt_%s_epoch_%d' %(network_label, epoch_label) 
        load_path = os.path.join(self._save_dir, load_filename)

        assert os.path.exists(load_path), 'Weights file not found. %s ' \
                  'Have you trained a model!? We are not providing one' % load_path
        optimizer.load_state_dict(load_path)
        print('loaded optimizer: %s' % load_path)

    def update_learning_rate(self, i_epoch):
        pass


    def _get_scheduler(self, scheduler_name):
        pass


    def print_network(self, network):
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()

        print(network)
        print('Total number of parameters: %d' % num_params)
