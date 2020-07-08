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

        elif network_name == 'G_NET':
            from .AttnGAN_nets import G_NET
            network = G_NET(*args, **kwargs)

        elif network_name == 'D_NET64':
            from .AttnGAN_nets import D_NET64
            network = D_NET64(*args, **kwargs)

        elif network_name == 'D_NET128':
            from .AttnGAN_nets import D_NET128
            network = D_NET128(*args, **kwargs)

        elif network_name == 'D_NET256':
            from .AttnGAN_nets import D_NET256
            network = D_NET256(*args, **kwargs)

        else:
            raise ValueError('Network [%s] not recognized!' % network_name)
    
        print('Network [%s] is created' % network_name)

        return network
