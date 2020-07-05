#!/usr/bin/env python
# -*- coding: utf-8 -*-

class NetworksFactory(object):

    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):

        if network_name == 'base':
            from .base_network import BaseNetwork
            network = BaseNetwork(*args, **kwargs)

        elif network_name == 'con2sen_encoder':
            from .con2sen_nets import Encoder
            network = Encoder(*args, **kwargs)

        elif network_name == 'con2sen_decoder':
            from .con2sen_nets import Decoder
            network = Decoder(*args, **kwargs)

        elif network_name == 'DAMSM_RNNEncoder':
            from .AttnGAN_nets import RNN_ENCODER
            network = RNN_ENCODER(*args, **kwargs)

        elif network_name == 'DAMSM_CNNEncoder':
            from .AttnGAN_nets import CNN_ENCODER
            network = CNN_ENCODER(*args, **kwargs)

        else:
            raise ValueError('Network [%s] not recognized!' % network_name)
    
        print('Network [%s] is created' % network_name)

        return network
