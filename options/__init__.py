#!/usr/bin/env python
# -*- coding: utf-8 -*-

class OptionFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(option_name):

        if option_name == 'base':
            from .base_options import BaseOptions
            option = BaseOptions()

        elif option_name == 'con2sen_train':
            from .con2sen_options import Con2SenOptions
            option = Con2SenOptions()

        elif option_name == 'con2sen_infer':
            from .con2sen_infer_options import Con2SenInferOptions
            option = Con2SenInferOptions()

        elif option_name == 'DAMSM':
            from .DAMSM_options import DAMSMOptions
            option = DAMSMOptions()

        elif option_name in ['VCD', 'GSC']:
            from .ut2i_options import UT2IOptions
            option = UT2IOptions()

        else:
            raise ValueError('Option [%s] not be recognized' % option_name)

        print('Option [%s] was created' % option_name)

        return option


