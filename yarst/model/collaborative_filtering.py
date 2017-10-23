from __future__ import division
from __future__ import print_function 

import pandas as pd
import numpy as np
import copy

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from ..utilities import MakeParamDict,EPSILON
from ..preprocessing import MinMaxScaler


def _get_similarity(x,y,metric='cosine',similarity_max_from_y=False, 
                    scale='nochange', num_threads = 1):
    
    distance = pairwise_distances(x, y, metric=metric, n_jobs=num_threads)
    distance[np.isnan(distance)] = distance.max()  
    
    if metric == 'cosine':
        sim = 1 - distance   
    else:
    # get the max value
        if similarity_max_from_y:
            scale_max = pairwise_distances(y, metric=metric,n_jobs=num_threads).max()
        else:
            scale_max = distance.max()        

        scaler = MinMaxScaler(feature_range=(0, scale_max),columns_wise=False)
        sim = scale_max - scaler.fit_transform(distance) 
        
    if scale == 'rescale':
        one_zero_scaler = MinMaxScaler(columns_wise = False)
        return one_zero_scaler.fit_transform(sim)
    elif scale == 'abs':
        return np.absolute(sim)
    elif scale == 'nochange':
        return sim

class Model_ColdStart:
    
    def __init__(self,rate_matrix,attribute_matrix):
        
        self.rate_matrix = rate_matrix
        self.attribute_matrix = attribute_matrix
        assert self.attribute_matrix is not None,'attribute data missed'
        
    def fit(self,x,y=None,
            attr_similarity_metric='cosine',
            attr_similarity_max_from_train = False,
            attr_similarity_scale = 'nochange', 
            num_threads = 1,verbose=1):
        
        assert attr_similarity_scale in ('nochange', 'rescale', 'abs')  
        
        if verbose:
            print('attr_similarity_metric:{},attr_similarity_max_from_train:{},attr_similarity_scale:{}'.format(attr_similarity_metric,attr_similarity_max_from_train,attr_similarity_scale))
            print('num_threads:{}'.format(num_threads))
        
        # input for similarity calculation
        attr_sim_input_x = x[2]
        attr_sim_input_y = self.attribute_matrix   
        
        self.sim_matrix = _get_similarity(
            attr_sim_input_x, attr_sim_input_y,
            attr_similarity_metric,
            attr_similarity_max_from_train, 
            attr_similarity_scale,
            num_threads)   
        
    def predict(self,x, predict_method='topk_wtd',k=50,verbose=1):
        
        assert predict_method in ('topk_wtd','topk_ave','baseline')
        if verbose:
            print('predict_method:{},k:{}'.format(predict_method,k))
        # get the signal of ratings at first
        rate_matrix_sign = self.rate_matrix.copy()
        rate_matrix_sign[self.rate_matrix != 0] = 1     
        
        prediction = []
        if predict_method == 'baseline': 
            denominator = self.sim_matrix.dot(rate_matrix_sign)
            denominator[denominator == 0] = EPSILON
            pred_all = self.sim_matrix.dot(self.rate_matrix) / denominator

            for u,i in np.column_stack(tuple(x[:2])):
                prediction.append(pred_all[u, i])        
        else:
            for u,i in np.column_stack(tuple(x[:2])):
                prediction.append(self._predict(
                    u,i,rate_matrix_sign,predict_method,k))  
        return prediction        
    
    def _predict(self,user,item,rate_matrix_sign,predict_method,k):
        
        top_k_neighbours = [np.argsort(self.sim_matrix[user, :])[:-k - 1:-1]]
        denominator = self.sim_matrix[user, :][top_k_neighbours].dot(
            rate_matrix_sign[:, item][top_k_neighbours])
        
        if denominator == 0:
            denominator = EPSILON       
        
        if predict_method == 'topk_wtd':
            pred = self.sim_matrix[user,:][top_k_neighbours].dot(
                self.rate_matrix[:, item][top_k_neighbours])
            output = pred / denominator   
            
        elif predict_method == 'topk_ave' :             
            ix_nonzero = ~(self.rate_matrix[:, item][top_k_neighbours]==0)
            if np.sum(ix_nonzero) == 0 : 
                output = 0
            else : 
                output = np.mean(self.rate_matrix[:, item][top_k_neighbours][ix_nonzero])                         
            
        return output            

class Model:
    
    def __init__(self,rate_matrix,attribute_matrix):
        
        self.rate_matrix = rate_matrix
        self.attribute_matrix = attribute_matrix
        if self.attribute_matrix is not None:
            self.use_attribute = True
        else:
            self.use_attribute = False
        
    def fit(self,x,y=None,
            similarity_type='rate',
            rate_similarity_metric='cosine',
            rate_similarity_max_from_train = False,
            rate_similarity_scale = 'nochange',
                
            attr_similarity_metric='cosine',
            attr_similarity_max_from_train = False,
            attr_similarity_scale = 'nochange', 
                
            similarity_weight = 0.5,
                
            use_bias = False,
            bias_type = 'user',
            num_threads = 1,
            verbose=1):
        """
        similarity_type: string
            rate: use rating only to calculate the similarity
            attribute: use attribute only to calculate the similarity
            hyrbid: use rate and attribute together to calculate the similarity
            
        rate_similarity_metric: string, how to calculate the similarity of rate
            ref: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html
        rate_similarity_max_from_train: boolean, whether get the max value of rate similarity from the train, or the max value is 1
        rate_similarity_scale: string, hwo to scale the similarity
            nochange: no change
            rescale: rescale similarity to [0,1]
            abs: absolute similarity    
        attr_similarity_metric: string, how to calculate the similarity of attribute
            ref: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html
        attr_similarity_max_from_train: boolean, whether get the max value of rate similarity from the train, or the max value is 1
        attr_similarity_scale: string, hwo to scale the similarity
            nochange: no change
            rescale: rescale similarity to [0,1]
            abs: absolute similarity
            
        similarity_weight: float, weight of similarity
        use_bias: boolean, use bias or not
        bias_type: string
            user: rating subtract the user based bias 
            item: rating subtract the item based bias
            hyrbid: ratings subtract item and user based bias
            zscore: rating subtract the user based bias and devided by user-based standard devation
        num_threads: int, how many cpus are used in calculating the similarity        
        """
        
        
        assert similarity_type in ('rate', 'attribute', 'hybrid'),'please set similarity correctly'
        if similarity_type in ('attribute', 'hybrid'):
            assert self.use_attribute,'attribute data missed'
        assert bias_type in ('user','item','hybrid','zscore')
        assert rate_similarity_scale in ('nochange', 'rescale', 'abs')
        assert attr_similarity_scale in ('nochange', 'rescale', 'abs')
        
        if verbose:
            print('similarity_type:{}'.format(similarity_type))
            print('rate_similarity_metric:{},'.format(rate_similarity_metric))
            print('rate_similarity_max_from_train:{}'.format(rate_similarity_max_from_train))
            print('rate_similarity_scale:{}'.format(rate_similarity_scale))
            print('attr_similarity_metric:{}'.format(attr_similarity_metric))
            print('attr_similarity_max_from_train:{}'.format(attr_similarity_max_from_train))
            print('attr_similarity_scale:{}'.format(attr_similarity_scale))
            print('similarity_weight:{}'.format(similarity_weight))
            print('use_bias:{},bias_type:{}'.format(use_bias,bias_type))
            print('num_threads:{}'.format(num_threads))
        
        self.use_bias = use_bias
        self.bias_type = bias_type
        
        # get encoded x
        self.fit_user_encoder = LabelEncoder()
        self.fit_encoded_user = self.fit_user_encoder.fit_transform(x[0]).tolist()  
        self.fit_user = x[0]
        
        # get bias
        if use_bias:
            rate_matrix_nan = self.rate_matrix.copy()
            self.user_bias = np.nanmean(rate_matrix_nan, axis=1)
            self.user_std = np.nanstd(rate_matrix_nan, axis=1)
            self.user_std[self.user_std == 0]= EPSILON
            
            self.item_bias = np.nanmean(rate_matrix_nan, axis=0)
            self.item_std = np.nanstd(rate_matrix_nan, axis=1)
            self.item_std[self.item_std == 0]= EPSILON
            
            # prepare the rate subract bias matrix
            if bias_type == 'user':
                self.rate_matrix_withbias=(self.rate_matrix-self.user_bias[:, np.newaxis]).copy()

            elif bias_type == 'item':
                self.rate_matrix_withbias=(self.rate_matrix-self.item_bias[np.newaxis, :]).copy()

            elif bias_type == 'hybrid':
                self.rate_matrix_withbias = (
                    self.rate_matrix-self.user_bias[:,np.newaxis]-self.item_bias[np.newaxis,:]).copy()

            elif bias_type == 'zscore':  
                self.rate_matrix_withbias=(self.rate_matrix-self.user_bias[:,np.newaxis])/self.user_std[:,np.newaxis].copy()        
            #ensure rating_bias_subtract and ratings have same 0 position
            self.rate_matrix_withbias[self.rate_matrix == 0] = 0
      
        
        # get the unique user and item 
        unique_users = np.unique(self.fit_encoded_user)
        unique_items = np.unique(x[1])
        
        # input for similarity calculation
        rate_sim_input_x = self.rate_matrix[unique_users, :]
        rate_sim_input_y = self.rate_matrix  
        if self.use_attribute:
            attr_sim_input_x = self.attribute_matrix[unique_users, :]
            attr_sim_input_y = self.attribute_matrix    
            
        # calculate the similarity
        if similarity_type == 'rate':
            sim_matrix = _get_similarity(
                rate_sim_input_x, rate_sim_input_y,
                rate_similarity_metric,
                rate_similarity_max_from_train, 
                rate_similarity_scale,
                num_threads)  
        elif similarity_type == 'attribute':
            sim_matrix = _get_similarity(
                attr_sim_input_x, attr_sim_input_y,
                attr_similarity_metric,
                attr_similarity_max_from_train, 
                attr_similarity_scale,
                num_threads)   
        elif similarity_type == 'hybrid':
            sim_matrix_rate = _get_similarity(
                rate_sim_input_x, rate_sim_input_y,
                rate_similarity_metric,
                rate_similarity_max_from_train, 
                rate_similarity_scale,
                num_threads)              
            sim_matrix_attr = _get_similarity(
                attr_sim_input_x, attr_sim_input_y,
                attr_similarity_metric,
                attr_similarity_max_from_train, 
                attr_similarity_scale,
                num_threads)   
            
            sim_matrix = sim_matrix_rate*similarity_weight + sim_matrix_attr*(1-similarity_weight)
        
        self.sim_matrix = sim_matrix        

        # udpate bias of fit population
        if use_bias:
            assert self.user_bias is not None
            assert self.item_bias is not None
            self.fit_user_bias = self.user_bias[unique_users]   
            self.fit_user_std = self.user_std[unique_users]    
                
        return self
            
        
    def predict(self,x,predict_method='topk_wtd',k = 50,verbose=1):
        
        """
        x is a list
        predict_method: string, hwo to get the prediction
            topk_wtd: seach the k nearest neighbours and get the predicitons weigthed by similarity 
            topk_ave: seach the k nearest neighbours and get the predicitons by average
            topk_avedev: user'mean + simple average of neighbours' centered rating
            topk_avezscore: user'mean +user std * simple average of neighbours z-score rating
            baseline_withbias: ratings are subtracted based on the bias_type, prediciton is weighted average of all users
            hyrbid: nobias for topk neighbours 
            baseline: predicitons is mean of all users
        k: int,
        
        """
        assert predict_method in ('topk_wtd','topk_ave','topk_avedev','topk_avezscore', 
                                  'baseline_withbias', 'hybrid', 'baseline')
        if not self.use_bias:
            assert predict_method in ('topk_wtd','topk_ave','baseline'),'please set use_bias=True for {}'.format(predict_method)
        
        diff = len(set(np.unique(x[0])) - set(np.unique(self.fit_user)))
        assert diff ==0,'x inlcude cold start users'
        
        if verbose:
            print('predict_method:{}'.format(predict_method))
            print('k:{}'.format(k))
        
        x_encoded = copy.copy(x)
        x_encoded[0] = self.fit_user_encoder.transform(x_encoded[0]).tolist()
        
        # get the signal of ratings at first
        rate_matrix_sign = self.rate_matrix.copy()
        rate_matrix_sign[self.rate_matrix != 0] = 1            
        
        prediction = []
        if predict_method == 'baseline': 
            denominator = self.sim_matrix.dot(rate_matrix_sign)
            denominator[denominator == 0] = EPSILON
            pred_all = self.sim_matrix.dot(self.rate_matrix) / denominator

            for u,i in np.column_stack(tuple(x_encoded)):
                prediction.append(pred_all[u, i])
                
        elif predict_method == 'baseline_withbias':
            denominator = self.sim_matrix.dot(rate_matrix_sign)
            denominator[denominator == 0] = EPSILON

            pred_all = self.sim_matrix.dot(self.rate_matrix_withbias)/denominator            
            
            if self.bias_type == 'user':
                pred_all += self.fit_user_bias[:, np.newaxis]

            elif self.pred_all == 'item':
                pred_array += self.item_bias[np.newaxis, :]

            elif self.bias_type == 'hybrid':
                pred_all += self.fit_user_bias[:,np.newaxis] + self.item_bias[np.newaxis,:]

            elif self.bias_type == 'zscore':
                pred_all *= np.diag(self.fit_user_std[:,np.newaxis])+self.fit_user_bias[:, np.newaxis]

            for u,i in np.column_stack(tuple(x_encoded)):
                prediction.append(pred_all[u, i])
        else:
            for u,i in np.column_stack(tuple(x_encoded)):
                prediction.append(self._predict(
                    u,i,rate_matrix_sign,predict_method,self.bias_type,k))
                
        return prediction
                
    def _predict(self,user,item,rate_matrix_sign,predict_method,bias_type,k):
        
        top_k_neighbours = [np.argsort(self.sim_matrix[user, :])[:-k - 1:-1]]
        denominator = self.sim_matrix[user, :][top_k_neighbours].dot(
            rate_matrix_sign[:, item][top_k_neighbours])
        
        if denominator == 0:
            denominator = EPSILON       
        
        if predict_method == 'topk_wtd':
            pred = self.sim_matrix[user,:][top_k_neighbours].dot(
                self.rate_matrix[:, item][top_k_neighbours])
            output = pred / denominator   
            
        elif predict_method == 'topk_ave' :             
            ix_nonzero = ~(self.rate_matrix[:, item][top_k_neighbours]==0)
            if np.sum(ix_nonzero) == 0 : 
                output = 0
            else : 
                output = np.mean(self.rate_matrix[:, item][top_k_neighbours][ix_nonzero])                 
                                
        elif predict_method == 'topk_avedev' :             
            ix_nonzero = ~(self.rate_matrix[:, item][top_k_neighbours]==0)
                
            if np.sum(ix_nonzero) == 0 : 
                output = self.fit_user_bias[user]
            else : 
                output = self.fit_user_bias[user]+\
                    np.mean(self.rate_matrix[:, item][top_k_neighbours][ix_nonzero]-\
                    self.user_bias[top_k_neighbours][ix_nonzero]) 
               
        elif predict_method == 'topk_avezscore' :   
            ix_nonzero = ~(self.rate_matrix[:, item][top_k_neighbours]==0)    
            if np.sum(ix_nonzero) == 0 : 
                output = self.fit_user_bias[user]
            else :
                output = self.fit_user_bias[user]+\
                    self.fit_user_std[user] * np.mean((self.rate_matrix[:, item][top_k_neighbours][ix_nonzero]-\
                    self.user_bias[top_k_neighbours][ix_nonzero])/self.user_std[top_k_neighbours][ix_nonzero])            
            
        elif predict_method == 'hybrid': 
            
            pred = self.sim_matrix[user, :][top_k_neighbours].dot(
                self.rate_matrix_withbias[:, item][top_k_neighbours])
            pred = pred / denominator
            
            if bias_type == 'user':
                output = pred + self.fit_user_bias[user]

            elif bias_type == 'item':
                output = pred + self.item_bias[item]

            elif bias_type == 'hybrid':
                output = pred + self.fit_user_bias[user] + self.item_bias[item]

            elif bias_type == 'zscore':
                output = pred * self.fit_user_std[user] + self.fit_user_bias[user]            
            
        return output            

class CollaborativeFiltering:

    def __init__(self,use_attribute=False,cold_start=False):
        
        self.use_attribute = use_attribute
        self.cold_start = cold_start
        
    def build_model(self):
        
        if self.cold_start:
            return Model_ColdStart(self.rate_matrix,self.attribute_matrix)
        else:
            return Model(self.rate_matrix,self.attribute_matrix)
        
    def fit(self,df,x_col,y_col):
        """
        df: pandas dataframe
        x_col: dict
        y_col: dict
        """
        assert isinstance(df, pd.DataFrame) 
        self.fit_shape_ = df.shape
        
        x_col_default = {
            'user': None, 
            'item': None,
            'attribute':None}
        y_col_default = {'rate':None}        
        
        x_col_dict = MakeParamDict(x_col, x_col_default)
        y_col_dict = MakeParamDict(y_col, y_col_default)
        
        df_copy = df.copy()
        df_copy.loc[df_copy[y_col_dict['rate']]==0, y_col_dict['rate']]= EPSILON

        rate_matrix = df_copy.pivot(index=x_col_dict['user'],columns=x_col_dict['item'],
                                    values=y_col_dict['rate']).values
        rate_matrix[np.isnan(rate_matrix)] = 0
        self.rate_matrix = rate_matrix
        
        self.attribute_matrix = None
        if self.use_attribute:
            assert x_col_dict['attribute'] is not None,'please set attribute in x_col'  
            attribute_matrix = df_copy.sort_values(x_col_dict['user']).reset_index(drop=True)
            attribute_matrix = attribute_matrix.drop_duplicates(subset=x_col_dict['user']).reset_index(drop=True)
            self.attribute_matrix = attribute_matrix[x_col_dict['attribute']].values             

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
        
        model_input_x = [df[x_col_dict['user']].values,
                         df[x_col_dict['item']].values]
        
        if self.cold_start:
            attribute_matrix = df.sort_values(x_col_dict['user']).reset_index(drop=True)
            attribute_matrix = attribute_matrix.drop_duplicates(subset=x_col_dict['user']).reset_index(drop=True)  
            model_input_x += [attribute_matrix[x_col_dict['attribute']].values] 

        if return_y:
            model_input_y = df[y_col_dict['rate']].values
            return model_input_x,model_input_y 
        else:
            return model_input_x         
