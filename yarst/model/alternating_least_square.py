from __future__ import division
from __future__ import print_function 

from tqdm import tqdm
import pandas as pd
import numpy as np
from numpy.linalg import solve
from ..utilities import MakeParamDict,EPSILON

class Model:
    
    def __init__(self,user_weights,item_weights):
        
        self.user_weights = user_weights
        self.item_weights = item_weights
        
    def _predict(self,user,item): 
        return self.user_weights[user,:].dot(self.item_weights[item,:].T)
                
    def _fit(self,rate_matrix,reg_user=0.01,reg_item=0.01,):
        
        # update user weights
        iTi = self.item_weights.T.dot(self.item_weights)
        lambda_i = np.eye(iTi.shape[0])*reg_user
        for u in range(self.user_weights.shape[0]):
            self.user_weights[u,:] = solve((iTi + lambda_i),rate_matrix[u,:].dot(self.item_weights))
        
        uTu = self.user_weights.T.dot(self.user_weights)
        lambda_u = np.eye(uTu.shape[0]) * reg_item
        for i in range(self.item_weights.shape[0]):
            self.item_weights[i,:] = solve((uTu + lambda_u),rate_matrix[:,i].T.dot(self.user_weights))
                
    def fit(self,x,y,epochs=10,reg_user=0.01,reg_item=0.01,verbose=0):
        
        if verbose:
            print('epochs:{},reg_user:{},reg_item:{}'.format(epochs,reg_user,reg_item))
        
        df_rate = pd.DataFrame({'user':x[0],'item':x[1],'rate':y})
        df_rate.loc[df_rate['rate']==0, 'rate'] = EPSILON
        rate_matrix = df_rate.pivot(index='user',columns='item',values='rate')
        rate_matrix = rate_matrix.fillna(0)
        rate_matrix = rate_matrix.values
        
        if verbose:
            for _ in tqdm(range(epochs)):
                self._fit(rate_matrix,reg_user,reg_item)    
        else:
            for _ in range(epochs):
                self._fit(rate_matrix,reg_user,reg_item)   
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
    
    
class AlternatingLeastSquare:
    
    def __init__(self,components=100):
        
        self.components = components
        
    def build_model(self):
        
        model_params = self.model_params
        
        user_weights = np.random.random((model_params['user_num'], self.components))
        item_weights = np.random.random((model_params['item_num'], self.components))
        
        return Model(user_weights,item_weights)
    
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
            'item_num':len(np.unique(df[x_col_dict['item']].values))}       

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
        
        
