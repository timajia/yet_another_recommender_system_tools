from __future__ import division
from __future__ import print_function 

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

def _check_x(x):
    for key in ["user","item"]:
        assert key in x.keys(),"x miss key({})".format(key)  

class SGD(BaseEstimator):
    
    def __init__(self,
                 num_hidden_factor,
                 num_user,
                 num_item,
                 epochs = 10,
                 learning_rate = 0.001,
                 shuffle = True,
                 reg_user = 0.01,
                 reg_item = 0.01,
                 reg_bias_user = 0.01,
                 reg_bias_item = 0.01,
                 cold_start = False,
                 epsilon = 1e-7,
                 verbose = True
                ):
        
        self.num_hidden_factor = num_hidden_factor
        self.num_user = num_user
        self.num_item = num_item
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.shuffle = shuffle
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.reg_bias_user = reg_bias_user
        self.reg_bias_item = reg_bias_item
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
              
        self.user_weights_ = np.random.normal(
            scale = 1./self.num_hidden_factor,
            size = (num_user,self.num_hidden_factor))
            
        self.item_weights_ = np.random.normal(
            scale = 1./self.num_hidden_factor,
            size = (num_item,self.num_hidden_factor))  
            
        self.user_bias_ = np.zeros(num_user)
        self.item_bias_ = np.zeros(num_item)           
        
        self.global_bias_ = 0
        
    def _predict(self,user,item):
        prediction = self.user_bias_[user]+self.item_bias_[item]+self.global_bias_  
        prediction += self.user_weights_[user,:].dot(self.item_weights_[item,:].T)
        return prediction        
    
    def _train(self,array_rate):
        
        lr = self.learning_rate 
        row_ids, col_ids = array_rate.nonzero()
        training_indices = np.arange(len(row_ids))  
        if self.shuffle:
            np.random.shuffle(training_indices)            
           
        for ix in training_indices:
            u = row_ids[ix]
            i = col_ids[ix]
            
            # error
            error = array_rate[u, i] - self._predict(u,i)
            
            # Update biases
            self.user_bias_[u] += lr*(error - self.reg_bias_user * self.user_bias_[u])
            self.item_bias_[i] += lr*(error - self.reg_bias_item * self.item_bias_[i]) 
            
            # Update weights
            self.user_weights_[u,:] += lr*(
                error*self.item_weights_[i,:] - self.reg_user * self.user_weights_[u,:])
            self.item_weights_[i,:] += lr*(
                error*self.user_weights_[u,:] - self.reg_item * self.item_weights_[i,:])        

    def train(self,x,y):
        assert isinstance(x,dict)
        _check_x(x)
            
        df = pd.DataFrame({"user":x["user"],"item":x["item"],"rate":y})
        df.loc[df['rate']==0, 'rate'] = self.epsilon
        
        df_rate = df.pivot(index='user',columns='item',values='rate')
        df_rate = df_rate.fillna(0)
        array_rate = df_rate.values      
        
        #update global_bias
        self.global_bias_ = np.mean(y)
        
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
    
