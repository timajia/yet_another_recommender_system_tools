from __future__ import division
from __future__ import print_function 

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler

class MinMaxScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, feature_range = (0,1), columns_wise = True, copy = True):
        """
        similari to MinMaxScaler(sklearn)
        just add one more parameter
        columns_wise: bool
            use column wise mean or not
            if False, then the min and max is based on entire matrix
        ref: 
        """
        
        self.feature_range = feature_range    
        self.columns_wise = columns_wise
        self.copy = copy
        
    def fit(self, x, y=None):
        
        x = check_array(x)
        self.input_shape_ = x.shape        
        
        if self.columns_wise:
            self.scaler = SKMinMaxScaler(feature_range=self.feature_range,copy=self.copy)
            self.scaler.fit(x,y)
            
        else:
            self.x_min = x.min()
            self.x_max = x.max()
            
            self.feature_min = self.feature_range[0]
            self.feature_max = self.feature_range[1]
        return self
        
    def transform(self,x):
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])

        # Input validation
        x = check_array(x)
        
        if x.shape != self.input_shape_:
            raise ValueError("Shape of input is different from what was seen in fit")   
        
        if self.columns_wise:
            return self.scaler.transform(x)
        else:
            x_std = (x - self.x_min) / (self.x_max - self.x_min)        
            return x_std * (self.feature_max - self.feature_min) + self.feature_min
 

    def inverse_transform(self,x):
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])

        # Input validation
        x = check_array(x)
        
        if x.shape != self.input_shape_:
            raise ValueError("Shape of input is different from what was seen in fit")   
        
        if self.columns_wise:
            return self.scaler.inverse_transform(x)
        else:
            x_std = (x - self.feature_min) / (self.feature_max - self.feature_min)
            
            return x_std * (self.x_max - self.x_min) + self.x_min    
        
