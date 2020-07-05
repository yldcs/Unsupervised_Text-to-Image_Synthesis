from .base_model import BaseModel
from networks import NetworksFactory
from datasets import CustomDatasetDataLoader
import os 
import json
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from utils.beam_search import beam_decode
from collections import OrderedDict
from utils.util import idxs2sentences
import sys
import _pickle as pkl

class Con2SenTrain(BaseModel):

    def __init__(self, opt):
        super(Con2SenTrain, self).__init__(opt)
        """
        Initialize trainer.
        """
        self._opt = opt
        self._build_models()
        self._init_train_vars()
        self._init_loss()
        
    def _build_models(self):

        self._enc = NetworksFactory.get_by_name('con2sen_encoder', self._opt)
        self._dec = NetworksFactory.get_by_name('con2sen_decoder', self._opt)
        self._enc.cuda()
        self._dec.cuda()

    def _init_train_vars(self):

        self._enc_optimizer = torch.optim.Adam(self._enc.parameters(), lr=self._opt.learning_rate)
        self._dec_optimizer = torch.optim.Adam(self._dec.parameters(), lr=self._opt.learning_rate)
    
    def _init_loss(self):
        
        self._crt_ce = nn.CrossEntropyLoss(ignore_index=0)

        self._loss_ce = self._Tensor([0])

    def set_input(self, input):
       
        concept, con_len = input['concepts'], input['concept_lens']
        sentence, sent_len = input['captions'], input['caption_lens']
        
        concept, sentence = concept.transpose(1, 0), sentence.transpose(1, 0)
        self._sentence = sentence
        self._concept = concept.cuda()
        self._con_len = con_len.cuda()
        self._sent_in = sentence[:-1].cuda()
        self._target = sentence[1:].contiguous().view(-1).cuda()

    def set_train(self):
        self._enc.train()
        self._dec.train()

    def set_eval(self):
        self._enc.eval()
        self._dec.eval()

    def optimize_parameters(self):
        """
        Update parameters.
        """

        out = self.forward()
        
        out = out.view(-1, out.size(2)).contiguous()
        self._loss_ce = self._crt_ce(out, self._target)

        # optimizer zero grad
        self._enc_optimizer.zero_grad()
        self._dec_optimizer.zero_grad()

        # backward
        self._loss_ce.backward()

        # clip gradients
        clip_grad_norm_(self._enc.parameters(), self._opt.grad_clip)
        clip_grad_norm_(self._dec.parameters(), self._opt.grad_clip)

        # optimizer step
        self._enc_optimizer.step()
        self._dec_optimizer.step()

    def forward(self):
        
        hidden = self._enc(self._concept, self._con_len)
        cell = torch.zeros_like(hidden)
        out = self._dec(self._sent_in, hidden, cell)

        self._hidden = hidden
        self._cell = cell
        return out

    def _display_current_results(self):
         
        gens, _ = beam_decode(self._dec, 3, (self._hidden, self._cell))
        gens = idxs2sentences(gens.tolist(), self._vocab)
        sens = idxs2sentences(self._sentence.transpose(1, 0)[:3].tolist(), self._vocab)
        cons = idxs2sentences(self._concept.transpose(1, 0)[:3].tolist(), self._vocab)

        for i in range(1):
            print('con', cons[i])
            print('sen', sens[i])
            print('gen', gens[i])

    def _display_current_errors(self):

        errors = {}
        errors['loss_ce'] = self._loss_ce.item()
        print('loss_ce: %.4f' % errors['loss_ce'])

    def display_terminal(self):
        self._display_current_errors()
        self._display_current_results()

    def save(self, label):
        """
        Save the models periodically.
        """
        self._save_network(self._enc, 'Enc', label)
        self._save_network(self._dec, 'Dec', label)
        
        self._save_optimizer(self._enc, 'Enc', label)
        self._save_optimizer(self._dec, 'Dec', label)

    def load(self):
        load_epoch = self._opt.load_epoch
        self._load_network(self._enc, 'Enc', load_epoch)
        self._load_network(self._dec, 'Dec', load_epoch)

        if self.is_train:
            self._load_optimizer(self._enc_optimizer, 'Enc', load_epoch)
            self._load_optimizer(self._dec_optimizer, 'Dec', load_epoch)



