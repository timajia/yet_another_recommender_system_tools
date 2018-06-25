from __future__ import division
from __future__ import print_function 

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from numpy.linalg import solve

def _check_x(x):
    for key in ["user","item"]:
        assert key in x.keys(),"x miss key({})".format(key)  

class ALS(BaseEstimator):
    
    def __init__(self,
                 num_hidden_factor,
                 num_user,
                 num_item,
                 epochs = 10,
                 reg_user = 0.01,
                 reg_item = 0.01,
                 cold_start = False,
                 epsilon = 1e-7,
                 verbose = True
                ):
        self.num_hidden_factor = num_hidden_factor
        self.num_user = num_user
        self.num_item = num_item
        self.epochs = epochs
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.cold_start = cold_start
        self.epsilon = epsilon
        self.verbose = verbose
        
        self._initial_weights()
        
    def _initial_weights(self):
        
        num_user = self.num_user 
        num_item = self.num_item    
            
        if self.cold_start:
            num_user += 1
            num_item += 1  
              
        self.user_weights_ = np.random.random((num_user,self.num_hidden_factor))
            
        self.item_weights_ = np.random.random((num_item,self.num_hidden_factor))  
        
    def _predict(self,user,item):
        return self.user_weights_[user,:].dot(self.item_weights_[item,:].T)   
    
    def _train(self,array_rate):
        
        # update user weights
        iTi = self.item_weights_.T.dot(self.item_weights_)
        lambda_i = np.eye(iTi.shape[0]) * self.reg_user
        for u in range(self.user_weights_.shape[0]):
            self.user_weights_[u,:] = solve(
                (iTi + lambda_i),array_rate[u,:].dot(self.item_weights_))
            
        #update item weights
        uTu = self.user_weights_.T.dot(self.user_weights_)
        lambda_u = np.eye(uTu.shape[0]) * self.reg_item
        for i in range(self.item_weights_.shape[0]):
            self.item_weights_[i,:] = solve(
                (uTu + lambda_u),array_rate[:,i].T.dot(self.user_weights_))      

    def train(self,x,y):
        assert isinstance(x,dict)
        _check_x(x)
            
        df = pd.DataFrame({"user":x["user"],"item":x["item"],"rate":y})
        df.loc[df['rate']==0, 'rate'] = self.epsilon
        
        df_rate = df.pivot(index='user',columns='item',values='rate')
        df_rate = df_rate.fillna(0)
        array_rate = df_rate.values      
        
        if self.verbose:
            for _ in tqdm(range(self.epochs)):
                self._train(array_rate)  
                
        else:
            for _ in range(self.epochs):
                self._fit(array_rate)       
    
    def predict(self,x):
        assert isinstance(x,dict)
        _check_x(x)
        
        users = x["user"]
        items = x["item"]
        if self.cold_start:
            users = np.clips(users,0,self.num_user+1)
            items = np.clips(items,0,self.num_item+1)
        
        predictions = []
        for idx in range(len(x["user"])):
            predictions.append(self._predict(users[idx],items[idx]))
        return predictions    
    
