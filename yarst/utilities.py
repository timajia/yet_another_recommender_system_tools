from __future__ import division
from __future__ import print_function 

import numpy as np
import copy

EPSILON = 10e-8

def MakeParamDict(input_dict,default_dict):
    
    if input_dict is not None:
        diff = list(set(input_dict.keys())-set(default_dict.keys()))
        assert len(diff) == 0,'These keys {} are not set correctly(not used)'.format(diff)    
    
    if input_dict is None:
        output_dict = default_dict
    else:
        #output_dict = copy.deepcopy(input_dict)
        output_dict = input_dict
        
        for key in default_dict.keys():
            if key not in output_dict:
                output_dict[key] = default_dict[key]     
                
    return output_dict
    
