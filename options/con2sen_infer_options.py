from .base_options import BaseOptions

class Con2SenInferOptions(BaseOptions):

    def initialize(self):

        BaseOptions.initialize(self)

        #-------model  config -------
        #-------t2t config ---------------
        self._parser.add_argument('--input_size',type=int,default= 512)
        self._parser.add_argument('--hidden_size',type=int,default=512)
        self._parser.add_argument('--dropout',type=float,default=0.2)

        #-------optimizer config -------------------------

        self._parser.add_argument('--learning_rate', type=float,default=1e-3)

        self._parser.add_argument('--save_path', type=str,
                                  default='./data/coco/pseudo/pseudos.pkl')



