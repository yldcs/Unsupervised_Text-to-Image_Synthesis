import argparse
import os
from utils import util

class BaseOptions(object): 

    def __init__(self):

        self._parser = argparse.ArgumentParser()
        self._initialized = False
        self.is_train = False

    def initialize(self):

        # device config
        self._parser.add_argument('--gpu_ids', type=str, default='0')
        self._parser.add_argument('--CUDA', action='store_false')

        # factory name
        self._parser.add_argument('--option_name', type=str, default='base')
        self._parser.add_argument('--dataset_name', type=str, default='base')
        self._parser.add_argument('--model_name', type=str, default='base')
        self._parser.add_argument('--network_name', type=str, default='base')
        self._parser.add_argument('--scheduler_name', type=str, default='step')
        self._parser.add_argument('--G_name', type=str, default='base')
        self._parser.add_argument('--D_name', type=str, default='base')

        # directory/file name
        self._parser.add_argument('--data_dir', type=str, default='./data')
        self._parser.add_argument('--outputs_dir', type=str, default='./outputs')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./outputs/checkpoints')
        self._parser.add_argument('--ckpt_name', type=str, default='ckpt')
        self._parser.add_argument('--load_path', type=str, default='None')
        self._parser.add_argument('--load_label', type=str, default='epoch_-1')
        self._parser.add_argument('--load_epoch', type=int, default='-1')


        # dataloader config
        self._parser.add_argument('--batch_size', type=int, default=64)
        self._parser.add_argument('--num_workers_train', type=int, default=4)
        self._parser.add_argument('--num_workers_test', type=int, default=4)
        self._parser.add_argument('--drop_last', action='store_false')

        # vocabulary config
        self._parser.add_argument('--vocab_size', type=int, default=5535)
        self._parser.add_argument('--word_count', type=str, default='coco_word_counts.txt')

        self._initialized = True
        
    def parse(self):
        
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        self._get_set_gpus()
        self._opt.is_train = self.is_train
        args = vars(self._opt)
        self._save(args)

        self._set_check_load_path()
        self._print(args)


        return self._opt

    def _get_set_gpus(self):
        
        os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'

        if len(self._opt.gpu_ids) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = self._opt.gpu_ids
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "0"

        self._opt.gpu_ids = [int(i) for i in self._opt.gpu_ids.split(',')]

    def _print(self, args):
        print('---------------- Options ---------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('---------------- End --------------------')

    def _set_check_load_path(self):

        load_path = self._opt.load_path
        if load_path != 'None':
            assert os.path.exists(load_path), "We don't find %s" % load_path
            self._opt.checkpoints_dir = os.path.dirname(load_path)
            filename = os.path.basename(load_path)
            filename = filename.split('.')[0].split('_')[-2:]
            load_label = "_".join(filename)
            self._opt.load_label = load_label

            if 'epoch' in load_label:
                self._opt.load_epoch = int(filename[-1].split('.')[0])

    def _save(self, args):

        expr_dir = os.path.join(self._opt.outputs_dir, 'checkpoints', self._opt.ckpt_name)
        print(expr_dir)
        util.mkdir(expr_dir)
        self._opt.checkpoints_dir = expr_dir
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('--------- Options ---------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------- End -------------\n')
