from __future__ import division
from __future__ import print_function 

import numpy as np
import pandas as pd
import copy
from sklearn.externals import joblib
from ..utilities import MakeParamDict


class Recommender:
    
    def __init__(self,
                 estimator,
                 model_file_path = None,
                 verbose = 1
                ):
        
        self.estimator = estimator
            
        if model_file_path is not None:
            self.model = joblib.load(model_file_path)
            if verbose:
                print('model built via model file at {}'.format(model_file_path))
            
        else:
            self.model = estimator.build_model()
            if verbose:
                print('model built via {}'.format(estimator.__class__.__name__))
        
    def fit(self,x,**kwargs):
        assert isinstance(x, pd.DataFrame)
        
        model_input_x,model_input_y = self.estimator.transform(x,return_y=True)
        
        self.model.fit(model_input_x,model_input_y,**kwargs)
                
        return self
    
        
    def predict(self,x,keep_ids=None,use_clip=False,clip_range=[0,1],**kwargs):
        
        train_input = self.estimator.transform(x,return_y=False)
        
        prediction = self.model.predict(train_input,**kwargs)
        if use_clip:
            prediction = np.clip(prediction,clip_range[0], clip_range[1])        
        
        if keep_ids is not None:
            return pd.concat([x[keep_ids],
                              pd.DataFrame(prediction,columns=['prediction'])],axis=1)
        else:
            return pd.concat([x,pd.DataFrame(prediction,columns=['prediction'])],axis=1)

    def evaluate(self,x,metrics,**kwargs):
        
        y_pred = self.predict(x,**kwargs)['prediction'].values
        _,y_true = self.estimator.transform(x,return_y=True)
        
        eval_out = {}
        for fn in metrics:
            eval_out[fn.__name__] = fn(y_true,y_pred)  
        return eval_out    
