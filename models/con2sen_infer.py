from .base_model import BaseModel
from utils.util import write_pickle_file
from networks import NetworksFactory
import os 
import json
import torch
from torch.nn.utils import clip_grad_norm_
from utils.beam_search import beam_decode
from utils.util import idxs2sentences

class Con2SenInfer(BaseModel):

    def __init__(self, opt):
        super(Con2SenInfer, self).__init__(opt)
        """
        Initialize trainer.
        """

        self._opt = opt
        self._build_models()
        self.info = {}
        
    def _build_models(self):

        self._enc = NetworksFactory.get_by_name('con2sen_encoder', self._opt)
        self._dec = NetworksFactory.get_by_name('con2sen_decoder', self._opt)
        self._enc.cuda()
        self._dec.cuda()

    def set_input(self, input):
        
        
        concepts, concept_lens = input['concepts'], input['concept_lens']
        concepts = concepts.transpose(1, 0)

        self._keys = input['key']
        self._concepts = concepts.cuda()
        self._concept_lens = concept_lens.cuda()

    def set_eval(self):

        self._enc.eval()
        self._dec.eval()

    def forward(self):
        
        hidden = self._enc(self._concepts, self._concept_lens)
        cell = torch.zeros_like(hidden)
        gens, _ = beam_decode(self._dec, self._concepts.size(1), (hidden, cell))
        gens = idxs2sentences(gens.tolist(), self._vocab)
        cons = idxs2sentences(self._concepts.transpose(1, 0).tolist(), self._vocab)

        for i, key in enumerate(self._keys):
            self.info[key] = [gens[i], cons[i]]
        self._gens = gens
        self._cons = cons

    def display_terminal(self,i_train_batch, total_batches, elasped):
        print('[%s/%s] time: %.3f count: %d'\
              % (i_train_batch, total_batches, elasped, len(list(self.info.keys()))))
        print('key: ', self._keys[0])
        print('con: ', self._cons[0])
        print('gen: ', self._gens[0])

    def load(self):

        load_label = self._opt.load_label
        self._load_network(self._enc, 'Enc', load_label)
        self._load_network(self._dec, 'Dec', load_label)

