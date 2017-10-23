from __future__ import division
from __future__ import print_function 

from tqdm import tqdm
import pandas as pd
import numpy as np
from ..utilities import MakeParamDict,EPSILON

class Model:
    
    def __init__(self,user_weights,item_weights,user_bias,item_bias,global_bias):
        
        self.user_weights = user_weights
        self.item_weights = item_weights
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.global_bias = global_bias
        
    def _predict(self,user,item):
        prediction = self.user_bias[user]+self.item_bias[item]+self.global_bias  
        prediction += self.user_weights[user,:].dot(self.item_weights[item,:].T)
        
        return prediction
                
    
    def _fit(self,rate_matrix,learning_rate,shuffle,
             reg_user=0.01,reg_item=0.01,reg_bias_user=0.01,reg_bias_item=0.01):
        
        lr = learning_rate 
        row_ids, col_ids = rate_matrix.nonzero()
        training_indices = np.arange(len(row_ids))  
        if shuffle:
            np.random.shuffle(training_indices)            
           
        for ix in training_indices:
            u = row_ids[ix]
            i = col_ids[ix]
            
            # error
            error = rate_matrix[u, i] - self._predict(u,i)
            
            # Update biases
            self.user_bias[u] += lr*(error - reg_bias_user * self.user_bias[u])
            self.item_bias[i] += lr*(error - reg_bias_item * self.item_bias[i]) 
            
            # Update weights
            self.user_weights[u,:] += lr*(error*self.item_weights[i,:]-reg_user*self.user_weights[u,:])
            self.item_weights[i,:] += lr*(error*self.user_weights[u,:]-reg_item*self.item_weights[i,:])            
            
    
    def fit(self,x,y,epochs=10,learning_rate=0.001,shuffle=True,
            reg_user=0.01,reg_item=0.01,reg_bias_user=0.01,reg_bias_item=0.01,verbose=0):
        
        if verbose:
            print('epochs:{},learning_rate:{},shuffle:{}'.format(epochs,learning_rate,shuffle))
            print('reg_user:{},reg_item:{},reg_bias_user:{},reg_bias_item:{}'.format(
                reg_user,reg_item,reg_bias_user,reg_bias_item))
        
        df_rate = pd.DataFrame({'user':x[0],'item':x[1],'rate':y})
        df_rate.loc[df_rate['rate']==0, 'rate'] = EPSILON
        rate_matrix = df_rate.pivot(index='user',columns='item',values='rate')
        rate_matrix = rate_matrix.fillna(0)
        rate_matrix = rate_matrix.values
        
        if verbose:
            for _ in tqdm(range(epochs)):
                self._fit(rate_matrix,learning_rate,shuffle,
                          reg_user,reg_item,reg_bias_user,reg_bias_item)  
                
        else:
            for _ in range(epochs):
                self._fit(rate_matrix,learning_rate,shuffle,
                          reg_user,reg_item,reg_bias_user,reg_bias_item)   
                
        return self
 
        
    def predict(self,x):
        """
        x is user,item pairs, [[1,2],[2,3],[3,4]]
        """
        prediction = []
        x_array = np.column_stack(tuple(x)) 
        for u,i in x_array:
            prediction.append(self._predict(u,i))
        return prediction
    
    
class StochasticGradientDescent:
    
    def __init__(self,components=100):
        
        self.components = components
        
    def build_model(self):
        
        model_params = self.model_params
        
        user_weights = np.random.normal(
            scale=1./self.components,size=(model_params['user_num'],self.components))        
        item_weights = np.random.normal(
            scale=1./self.components,size=(model_params['item_num'],self.components))
        
        user_bias = np.zeros(model_params['user_num'])
        item_bias = np.zeros(model_params['item_num'])
        
        global_bias = model_params['global_bias']
        
        return Model(user_weights,item_weights,user_bias,item_bias,global_bias)
    
    def fit(self,df,x_col,y_col):
        assert isinstance(df, pd.DataFrame) 
        self.fit_shape_ = df.shape
        
        x_col_default = {
            'user': None, 
            'item': None}
        y_col_default = {'rate':None}
        
        x_col_dict = MakeParamDict(x_col, x_col_default)
        y_col_dict = MakeParamDict(y_col, y_col_default)
        
        model_params= {
            'user_num':len(np.unique(df[x_col_dict['user']].values)),
            'item_num':len(np.unique(df[x_col_dict['item']].values)),
            'global_bias': np.mean(df[y_col_dict['rate']].values)}       

        self.model_params = model_params
        self.x_col_dict = x_col_dict
        self.y_col_dict = y_col_dict   
        
        return self
    
    def transform(self,df,return_y=True):
        assert isinstance(df, pd.DataFrame)  
        try:
            fit_shape_ = self.fit_shape_
        except AttributeError:
            print({'{} does not fit'.format(self.__class__.__name__)})    
            
        x_col_dict = self.x_col_dict
        y_col_dict = self.y_col_dict
        
        model_input_x = [df[x_col_dict['user']].values,df[x_col_dict['item']].values]

        if return_y:
            model_input_y = df[y_col_dict['rate']].values
            return model_input_x,model_input_y 
        else:
            return model_input_x 
        
        
