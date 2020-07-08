from .base_model import BaseModel
from utils.util import weights_init
from networks import NetworksFactory
import torch
from copy import deepcopy
import os

class UT2ITrain(BaseModel):

    def __init__(self, opt):
        super(UT2ITrain, self).__init__(opt)
        """
        Initialize trainer.
        """
        self._opt = opt
        self._build_models()
        self._avg_param_G = self._copy_G_params(self._G)
        self._init_train_vars()
        self._init_loss()
        
    def _load_pretrain(self, net, path):

        assert path, 'we can not find pretrained file %s' % path
        state_dict = torch.load(path)
        net.load_state_dict(state_dict)
        print('load pretrain net from %s' % path)

    def _build_models(self):

        self._rnn = NetworksFactory.get_by_name('DAMSM_RNNEncoder', self._opt)
        self._load_pretrain(self._rnn, self._opt.NET_E)
        self._freeze(self._rnn)
        self._rnn.eval()
        self._cnn = NetworksFactory.get_by_name('DAMSM_CNNEncoder', self._opt)
        self._load_pretrain(self._cnn, self._opt.NET_E.replace('text', 'image'))
        self._freeze(self._cnn)
        self._cnn.eval()

        self._G = NetworksFactory.get_by_name('G_NET', self._opt)
        self._G.apply(weights_init)

        self._D = []
        self._D.append(NetworksFactory.get_by_name('D_NET64', self._opt))
        self._D.append(NetworksFactory.get_by_name('D_NET128', self._opt))
        self._D.append(NetworksFactory.get_by_name('D_NET256', self._opt))

        for i in range(3):
            self._D[i].apply(weights_init)

        if os.path.exist(self._opt.NET_G):
            self._load_pretrain(self._G, self._opt.NET_G)
            for i in range(3):
                self._load_pretrain(self._D[i],
                    self._opt.NET_G.replace('G', 'D%d'%i))

        self._rnn.cuda()
        self._cnn.cuda()
        self._G.cuda()
        for i in range(3):
            self._D[i].cuda()

    def _init_train_vars(self):

        self._optimizersD = []
        for i in range(len(self._D)):
            optimizer = torch.optim.Adam(self._D[i].parameters(),
                      lr=self._opt.DISCRIMINATOR_LR, betas=(0.5, 0.999))
            self._optimizersD.append(optimizer)

        self._optimizerG = torch.optim.Adam(self._G.parameters(),
                        lr=self._opt.GENERATOR_LR, betas=(0.5, 0.999))

    def _init_loss(self):
        
        self._s_loss0 = self._Tensor([0])
        self._s_loss1 = self._Tensor([0])
        self._w_loss0 = self._Tensor([0])
        self._w_loss1 = self._Tensor([0])
        self._loss = self._Tensor([0])

    def set_input(self, input):
       
        self._bs = self._opt.batch_size
        self._real_labels = torch.ones(self._bs).cuda()
        self._fake_labels = torch.zeros(self._bs).cuda()
        self._match_labels = torch.arange(self._bs).long().cuda()

        self._noise = torch.randn(self._bs, self._opt.Z_DIM).cuda()

        self._pseudos = input['pseudos'].cuda()
        self._pseudo_lens = input['pseudo_lens'].cuda()
        self._image_concepts = input['image_concepts'].cuda()
        self._corpus = input['corpus'].cuda()
        self._corpus_lens = input['corpus_lens'].cuda()
        self._corpus_concepts = input['corpus_concepts'].cuda()

        self._images = []

        for image in input['images']:
            self._images.append(image.cuda())

        self._class_ids = input['class_id'].numpy()
        self._keys = input['key']

    def set_train(self):
        self._rnn.train()
        self._cnn.train()

    def set_eval(self):
        self._rnn.eval()
        self._cnn.eval()


    def _gsc_optimize_parameters(self):
        """
        Update parameters.
        """
        from utils.gsc_losses import discriminator_loss, generator_loss, KL_loss
        hidden = self._rnn.init_hidden(self._bs)
        pseudo_words_embs, pseudo_sent_emb = self._rnn(self._corpus, self._corpus_lens, hidden)
        pseudo_words_embs, pseudo_sent_emb = pseudo_words_embs.detach(), pseudo_sent_emb.detach()

        corpus_words_embs, corpus_sent_emb = self._rnn(self._corpus, self._corpus_lens, hidden)
        corpus_words_embs, corpus_sent_emb = corpus_words_embs.detach(), corpus_sent_emb.detach()
        mask = (self._corpus == 0)
        num_words = corpus_words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]

        fake_imgs, _, mu, logvar = self._G(self._noise, corpus_sent_emb, corpus_words_embs, mask)

        self._errD_total = 0
        self._D_logs = ''

        for i in range(len(self._D)):
            self._D[i].zero_grad()
            errD = discriminator_loss(self._opt, self._D[i], self._images[i], fake_imgs[i],
                   pseudo_sent_emb, corpus_sent_emb,  self._real_labels, self._fake_labels,
                   self._image_concepts, self._corpus_concepts)

            errD.backward()
            self._optimizersD[i].step()
            self._errD_total += errD
            self._D_logs += 'errD%d: %.3f ' % (i, errD.item())

        # Update G network: maximize log(D(G(z)))

        self._G.zero_grad() 
        self._errG_total, self._G_logs = \
            generator_loss(self._opt, self._D, self._cnn, fake_imgs, self._real_labels, self._image_concepts,
                           corpus_words_embs, corpus_sent_emb, self._match_labels, self._corpus_lens, self._class_ids)

        kl_loss = KL_loss(mu, logvar)
        self._errG_total += kl_loss
        self._G_logs += 'kl_loss: %.2f ' % kl_loss.item()

        # backward and update parameters
        self._errG_total.backward()
        self._optimizerG.step()

        for p, avg_p in zip(self._G.parameters(), self._avg_param_G):
            avg_p.mul_(0.999).add_(0.001, p.data)

    def _vcd_optimize_parameters(self):
        """
        Update parameters.
        """
        from utils.vcd_losses import discriminator_loss, generator_loss, KL_loss

        hidden = self._rnn.init_hidden(self._bs)
        words_embs, sent_emb = self._rnn(self._pseudos, self._pseudo_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        mask = (self._pseudos == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]

        fake_imgs, _, mu, logvar = self._G(self._noise, sent_emb, words_embs, mask)

        self._errD_total = 0
        self._D_logs = ''

        for i in range(len(self._D)):
            self._D[i].zero_grad()
            errD = discriminator_loss(self._opt, self._D[i], self._images[i], fake_imgs[i],
                sent_emb, self._real_labels, self._fake_labels, self._image_concepts)
            errD.backward()
            self._optimizersD[i].step()
            self._errD_total += errD
            self._D_logs += 'errD%d: %.3f ' % (i, errD.item())

        # Update G network: maximize log(D(G(z)))

        self._G.zero_grad()
        self._errG_total, self._G_logs = \
            generator_loss(self._opt, self._D, self._cnn, fake_imgs, self._real_labels, self._image_concepts,
                           words_embs, sent_emb, self._match_labels, self._pseudo_lens, self._class_ids)

        kl_loss = KL_loss(mu, logvar)
        self._errG_total += kl_loss
        self._G_logs += 'kl_loss: %.2f ' % kl_loss.item()

        # backward and update parameters
        self._errG_total.backward()
        self._optimizerG.step()

        for p, avg_p in zip(self._G.parameters(), self._avg_param_G):
            avg_p.mul_(0.999).add_(0.001, p.data)

    def optimize_parameters(self):
        if 'VCD' in self._opt.model_name:
            self._vcd_optimize_parameters()

        elif 'GSC' in self._opt.model_name:
            self._gsc_optimize_parameters()

        else:
            raise ValueError('model trainer %s could not recognized' % self._opt.model_name)

    def display_terminal(self, epoch, total_epoch, i_batch, total_batch, elapsed):
        
        print('| epoch {:3d}/{:3d} | batches {:5d}/{:5d} |'
              'ms/batch {:5.2f} |'
              .format(epoch, total_epoch, i_batch, total_batch, elapsed))
        print(self._G_logs)
        print(self._D_logs)

    def _load_params(self, model, new_param):
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)

    def _copy_G_params(self, model):
        flatten = deepcopy(list(p.data for p in model.parameters()))
        return flatten

    def save(self, label):
        """
        Save the models periodically.
        """
        back_param = self._copy_G_params(self._G)
        self._load_params(self._G, self._avg_param_G)
        self._save_network(self._G, 'G', label)
        self._load_params(self._G, back_param)
        self._save_optimizer(self._optimizerG, 'G', label)

        for i in range(3):
            self._save_network(self._D[i], 'D%d' % i, label)
            self._save_optimizer(self._optimizersD[i], 'D%d' % i, label)
        
    def load(self):

        load_label = self._opt.load_label
        self._load_network(self._G, 'text_encoder', load_label)
        self._load_optimizer(self._optimizerG, 'G', load_label)

        for i in range(3):
            self._load_optimizer(self._optimizersD[i], 'D%d' % i, load_label)
            self._load_network(self._D[i], 'D%d'%i, load_label)
