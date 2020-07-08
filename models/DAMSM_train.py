from .base_model import BaseModel
from networks import NetworksFactory
import os
import torch
from torch.nn.utils import clip_grad_norm_
from utils.losses import words_loss, sent_loss

class DAMSMTrain(BaseModel):

    def __init__(self, opt):
        super(DAMSMTrain, self).__init__(opt)
        """
        Initialize trainer.
        """
        self._opt = opt
        self._build_models()
        self._init_train_vars()
        self._init_loss()
        
    def _build_models(self):

        self._rnn = NetworksFactory.get_by_name('DAMSM_RNNEncoder', self._opt)
        self._cnn = NetworksFactory.get_by_name('DAMSM_CNNEncoder', self._opt)
        self._rnn.cuda()
        self._cnn.cuda()

    def _zero_grad(self):
        self._rnn.zero_grad()
        self._cnn.zero_grad()

    def _init_train_vars(self):

        params = list(self._rnn.parameters())
        for v in self._cnn.parameters():
            if v.requires_grad:
                params.append(v)

        self._optimizer = torch.optim.Adam(params,
                      lr=self._opt.ENCODER_LR, betas=(0.5, 0.999))

    def _init_loss(self):
        
        self._s_loss0 = self._Tensor([0])
        self._s_loss1 = self._Tensor([0])
        self._w_loss0 = self._Tensor([0])
        self._w_loss1 = self._Tensor([0])
        self._loss = self._Tensor([0])

    def set_input(self, input):
       
        self._pseudos = input['pseudos'].cuda()
        self._pseudo_lens = input['pseudo_lens'].cuda()

        self._images = input['images'][-1].cuda()
        self._class_ids = input['class_id'].numpy()
        self._keys = input['key']
        self._labels = torch.arange(self._opt.batch_size).long().cuda()

    def set_train(self):
        self._rnn.train()
        self._cnn.train()

    def set_eval(self):
        self._rnn.eval()
        self._cnn.eval()

    def optimize_parameters(self):
        """
        Update parameters.
        """

        words_features, sent_code, words_emb, sent_emb = self.forward()
        self._w_loss0, self._w_loss1, attn_maps = words_loss(self._opt, words_features, 
                 words_emb, self._labels, self._pseudo_lens, self._class_ids, self._opt.batch_size)

        self._loss = self._w_loss0 + self._w_loss1
        self._s_loss0, self._s_loss1 = \
            sent_loss(self._opt, sent_code, sent_emb, self._labels, self._class_ids, self._opt.batch_size)
        self._loss = self._s_loss0 + self._s_loss1
        
        self._loss.backward()

        # `clip_grad_norm_` helps prevent
        # the explording gradient problem in RNNs / LSTMs
        clip_grad_norm_(self._rnn.parameters(), self._opt.RNN_GRAD_CLIP)

        self._optimizer.step()

    def forward(self):
       
        words_features, sent_code = self._cnn(self._images)
        nef, att_sze = words_features.size(1), words_features.size(2)
        
        hidden = self._rnn.init_hidden(self._opt.batch_size)
        words_emb, sent_emb = self._rnn(self._pseudos, self._pseudo_lens, hidden)
        
        return words_features, sent_code, words_emb, sent_emb

    def display_terminal(self, epoch, total_epoch, i_batch, total_batch, elapsed):
        
        print('| epoch {:3d}/{:3d} | batches {:5d}/{:5d} |'
              'ms/batch {:5.2f} |'
              's_loss: {:5.2f} {:5.2f} |'
              'w_loss: {:5.2f} {:5.2f} |'
              .format(epoch, total_epoch, i_batch, total_batch, elapsed,
                      self._s_loss0.item(), self._s_loss1.item(),
                      self._w_loss0.item(), self._w_loss1.item()))

    def save(self, label):
        """
        Save the models periodically.
        """
        self._save_network(self._rnn, 'text_encoder', label)
        self._save_network(self._cnn, 'image_encoder', label)
        self._save_optimizer(self._optimizer, 'DAMSM', label)

    def load(self):

        load_label = self._opt.load_label
        self._load_network(self._enc, 'text_encoder', load_label)
        self._load_network(self._dec, 'image_encoder', load_label)

        if self.is_train:
            self._load_optimizer(self._enc_optimizer, 'text_encoder', load_label)
            self._load_optimizer(self._dec_optimizer, 'image_encoder', load_label)
