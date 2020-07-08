#!/usr/bin/env python
# -*- coding: utf-8 -*-

class ModelsFactory(object):

    def __init__(self):
        pass
    
    @staticmethod
    def get_by_name(model_name, *args, **kwargs):

        if model_name == 'base':
            from .base_model import BaseModel
            model = BaseModel(*args, **kwargs) 

        elif model_name == 'con2sen_train':
            from .con2sen_train import Con2SenTrain
            model = Con2SenTrain(*args, **kwargs)

        elif model_name == 'con2sen_infer':
            from .con2sen_infer import Con2SenInfer
            model = Con2SenInfer(*args, **kwargs)

        elif model_name == 'DAMSM_train':
            from .DAMSM_train import DAMSMTrain
            model = DAMSMTrain(*args, **kwargs)

        elif model_name in ['VCD_train', 'GSC_train']:
            from .ut2i_train import UT2ITrain
            model = UT2ITrain(*args, **kwargs)
        else:
            raise ValueError('Model [%s] could not be recognized' % model_name)
        
        print('Model [%s] was created' % model_name)

        return model
