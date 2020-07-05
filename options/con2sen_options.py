from .base_options import BaseOptions

class Con2SenOptions(BaseOptions):

    def initialize(self):

        BaseOptions.initialize(self)

        #-------model  config -------
        #-------t2t config ---------------
        self._parser.add_argument('--input_size',type=int,default= 512)
        self._parser.add_argument('--hidden_size',type=int,default=512)
        self._parser.add_argument('--dropout',type=float,default=0.2)

        #-------optimizer config -------------------------

        self._parser.add_argument('--learning_rate', type=float,default=1e-3)

        #-------train-------------------------------------
        self._parser.add_argument('--max_it',type=int, default=100000)
        self._parser.add_argument('--total_epoch', type=int, default=50)
        self._parser.add_argument('--it_interval',type=int, default=5000)
        self._parser.add_argument('--epoch_interval', type=int, default=40)
        self._parser.add_argument('--grad_clip',type=float,default=10)



