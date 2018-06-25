from __future__ import division
from __future__ import print_function 

import pandas as pd
import numpy as np
import copy

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from ..env_setup import EPSILON

def _check_x(x,similarity_type):
    for key in ["user","item"]:
        assert key in x.keys(),"x miss key({})".format(key) 
    
    if similarity_type in ["attribute","hyrbid"]:
        assert "attribute" in x.keys(),"x miss key(attribute)"
        
        
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
    
class _ColdStartModel:
    
    def __init__(self,array_rate,array_attrs):
        
        self.array_rate = array_rate
        self.array_attrs = array_attrs
        assert self.array_attrs is not None,'attribute data missed'
        
    def fit(self,
            x,
            attr_similarity_metric='cosine',
            attr_similarity_max_from_train = False,
            attr_similarity_scale = 'nochange', 
            num_threads = 1
           ):
        """
        x: dict
        """
        
        assert attr_similarity_scale in ('nochange', 'rescale', 'abs')  
        assert "attribute" in x.keys()
        
        # input for similarity calculation
        attr_sim_input_x = x["attribute"]
        attr_sim_input_y = self.array_attrs   
        
        self.sim_matrix = _get_similarity(
            attr_sim_input_x, attr_sim_input_y,
            attr_similarity_metric,
            attr_similarity_max_from_train, 
            attr_similarity_scale,
            num_threads)   
        
    def predict(self,x, predict_method='topk_wtd',k=50):
        
        """
        args: 
            x: dict
        """
        
        assert predict_method in ('topk_wtd','topk_ave','baseline')
        # get the signal of ratings at first
        array_rate_sign = self.array_rate.copy()
        array_rate_sign[self.array_rate != 0] = 1     
        
        prediction = []
        if predict_method == 'baseline': 
            denominator = self.sim_matrix.dot(array_rate_sign)
            denominator[denominator == 0] = EPSILON
            pred_all = self.sim_matrix.dot(self.array_rate) / denominator

            for u,i in np.column_stack([x["user"],x["item"]]):
                prediction.append(pred_all[u, i])        
        else:
            for u,i in np.column_stack([x["user"],x["item"]]):
                prediction.append(self._predict(
                    u,i,array_rate_sign,predict_method,k))  
        return prediction        
    
    def _predict(self,user,item,array_rate_sign,predict_method,k):
        
        top_k_neighbours = [np.argsort(self.sim_matrix[user, :])[:-k - 1:-1]]
        denominator = self.sim_matrix[user, :][top_k_neighbours].dot(
            array_rate_sign[:, item][top_k_neighbours])
        
        if denominator == 0:
            denominator = EPSILON       
        
        if predict_method == 'topk_wtd':
            pred = self.sim_matrix[user,:][top_k_neighbours].dot(
                self.array_rate[:, item][top_k_neighbours])
            output = pred / denominator   
            
        elif predict_method == 'topk_ave' :             
            ix_nonzero = ~(self.array_rate[:, item][top_k_neighbours]==0)
            if np.sum(ix_nonzero) == 0 : 
                output = 0
            else : 
                output = np.mean(self.array_rate[:, item][top_k_neighbours][ix_nonzero])                         
            
        return output            

class _Model:
    
    def __init__(self,array_rate,array_attrs):
        
        self.array_rate = array_rate
        self.array_attrs = array_attrs
        if self.array_attrs is not None:
            self.use_attribute = True
        else:
            self.use_attribute = False
        
    def fit(self,x,
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
            num_threads = 1
           ):
        
        assert similarity_type in ('rate', 'attribute', 'hybrid'),'please set similarity correctly'
        if similarity_type in ('attribute', 'hybrid'):
            assert self.use_attribute,'attribute data missed'
        assert bias_type in ('user','item','hybrid','zscore')
        assert rate_similarity_scale in ('nochange', 'rescale', 'abs')
        assert attr_similarity_scale in ('nochange', 'rescale', 'abs')
        
        self.use_bias = use_bias
        self.bias_type = bias_type
        
        # get encoded x
        self.fit_user_encoder = LabelEncoder()
        self.fit_encoded_user = self.fit_user_encoder.fit_transform(x["user"]).tolist()  
        self.fit_user = x["user"]
        
        # get bias
        if use_bias:
            array_rate_nan = self.array_rate.copy()
            self.user_bias = np.nanmean(array_rate_nan, axis=1)
            self.user_std = np.nanstd(array_rate_nan, axis=1)
            self.user_std[self.user_std == 0]= EPSILON
            
            self.item_bias = np.nanmean(array_rate_nan, axis=0)
            self.item_std = np.nanstd(array_rate_nan, axis=1)
            self.item_std[self.item_std == 0]= EPSILON
            
            # prepare the rate subract bias matrix
            if bias_type == 'user':
                self.array_rate_withbias=(self.array_rate - self.user_bias[:, np.newaxis]).copy()

            elif bias_type == 'item':
                self.array_rate_withbias=(self.array_rate - self.item_bias[np.newaxis, :]).copy()

            elif bias_type == 'hybrid':
                self.array_rate_withbias = (
                    self.array_rate-self.user_bias[:,np.newaxis]-self.item_bias[np.newaxis,:]).copy()

            elif bias_type == 'zscore':  
                self.array_rate_withbias=(self.array_rate\
                    - self.user_bias[:,np.newaxis])/self.user_std[:,np.newaxis].copy()        
            #ensure rating_bias_subtract and ratings have same 0 position
            self.array_rate_withbias[self.array_rate == 0] = 0
      
        # get the unique user and item 
        unique_users = np.unique(self.fit_encoded_user)
        unique_items = np.unique(x["item"])
        
        # input for similarity calculation
        rate_sim_input_x = self.array_rate[unique_users, :]
        rate_sim_input_y = self.array_rate  
        if self.use_attribute:
            attr_sim_input_x = self.array_attrs[unique_users, :]
            attr_sim_input_y = self.array_attrs    
            
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
            
        
    def predict(self,x,predict_method='topk_wtd',k = 50):

        assert predict_method in ('topk_wtd','topk_ave','topk_avedev','topk_avezscore', 
                                  'baseline_withbias', 'hybrid', 'baseline')
        if not self.use_bias:
            assert predict_method in ('topk_wtd','topk_ave','baseline'),'please set use_bias=True for {}'.format(predict_method)
        
        diff = len(set(np.unique(x["user"])) - set(np.unique(self.fit_user)))
        assert diff ==0,'x inlcude cold start users'
        
        x_encoded = copy.copy(x)
        x_encoded["user"] = self.fit_user_encoder.transform(x_encoded["user"]).tolist()
        
        # get the signal of ratings at first
        array_rate_sign = self.array_rate.copy()
        array_rate_sign[self.array_rate != 0] = 1            
        
        prediction = []
        if predict_method == 'baseline': 
            denominator = self.sim_matrix.dot(array_rate_sign)
            denominator[denominator == 0] = EPSILON
            pred_all = self.sim_matrix.dot(self.array_rate) / denominator

            for u,i in np.column_stack([x_encoded["user"],x_encoded["item"]]):
                prediction.append(pred_all[u, i])
                
        elif predict_method == 'baseline_withbias':
            denominator = self.sim_matrix.dot(array_rate_sign)
            denominator[denominator == 0] = EPSILON

            pred_all = self.sim_matrix.dot(self.array_rate_withbias)/denominator            
            
            if self.bias_type == 'user':
                pred_all += self.fit_user_bias[:, np.newaxis]

            elif self.pred_all == 'item':
                pred_array += self.item_bias[np.newaxis, :]

            elif self.bias_type == 'hybrid':
                pred_all += self.fit_user_bias[:,np.newaxis] + self.item_bias[np.newaxis,:]

            elif self.bias_type == 'zscore':
                pred_all *= np.diag(self.fit_user_std[:,np.newaxis])+self.fit_user_bias[:, np.newaxis]

            for u,i in np.column_stack([x_encoded["user"],x_encoded["item"]]):
                prediction.append(pred_all[u, i])
        else:
            for u,i in np.column_stack([x_encoded["user"],x_encoded["item"]]):
                prediction.append(self._predict(
                    u,i,array_rate_sign,predict_method,self.bias_type,k))
                
        return prediction
                
    def _predict(self,user,item,array_rate_sign,predict_method,bias_type,k):
        
        top_k_neighbours = [np.argsort(self.sim_matrix[user, :])[:-k - 1:-1]]
        denominator = self.sim_matrix[user, :][top_k_neighbours].dot(
            array_rate_sign[:, item][top_k_neighbours])
        
        if denominator == 0:
            denominator = EPSILON       
        
        if predict_method == 'topk_wtd':
            pred = self.sim_matrix[user,:][top_k_neighbours].dot(
                self.array_rate[:, item][top_k_neighbours])
            output = pred / denominator   
            
        elif predict_method == 'topk_ave' :             
            ix_nonzero = ~(self.array_rate[:, item][top_k_neighbours]==0)
            if np.sum(ix_nonzero) == 0 : 
                output = 0
            else : 
                output = np.mean(self.array_rate[:, item][top_k_neighbours][ix_nonzero])                 
                                
        elif predict_method == 'topk_avedev' :             
            ix_nonzero = ~(self.array_rate[:, item][top_k_neighbours]==0)
                
            if np.sum(ix_nonzero) == 0 : 
                output = self.fit_user_bias[user]
            else : 
                output = self.fit_user_bias[user]+\
                    np.mean(self.array_rate[:, item][top_k_neighbours][ix_nonzero]-\
                    self.user_bias[top_k_neighbours][ix_nonzero]) 
               
        elif predict_method == 'topk_avezscore' :   
            ix_nonzero = ~(self.array_rate[:, item][top_k_neighbours]==0)    
            if np.sum(ix_nonzero) == 0 : 
                output = self.fit_user_bias[user]
            else :
                output = self.fit_user_bias[user]+\
                    self.fit_user_std[user] * np.mean((self.array_rate[:, item][top_k_neighbours][ix_nonzero]-\
                    self.user_bias[top_k_neighbours][ix_nonzero])/self.user_std[top_k_neighbours][ix_nonzero])            
            
        elif predict_method == 'hybrid': 
            
            pred = self.sim_matrix[user, :][top_k_neighbours].dot(
                self.array_rate_withbias[:, item][top_k_neighbours])
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

    
class UserBasedCollaborativeFiltering:
    
    def __init__(self,
                 similarity_type = "rate",
                 rate_similarity_metric='cosine',
                 rate_similarity_max_from_train = False,
                 rate_similarity_scale = 'nochange',  
                 
                 attr_similarity_metric='cosine',
                 attr_similarity_max_from_train = False,
                 attr_similarity_scale = 'nochange',      
                 
                 similarity_weight = 0.5,
                 use_bias = False,
                 bias_type = 'user',
                 
                 predict_method = "topk_wtd",
                 k =50,
                 n_jobs = 1,
                 
                 cold_start = False,
                 verbose = True
                ):
        
        
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
        
        predict_method: string, hwo to get the prediction
            topk_wtd: seach the k nearest neighbours and get the predicitons weigthed by similarity 
            topk_ave: seach the k nearest neighbours and get the predicitons by average
            topk_avedev: user'mean + simple average of neighbours' centered rating
            topk_avezscore: user'mean +user std * simple average of neighbours z-score rating
            baseline_withbias: ratings are subtracted based on the bias_type, prediciton is weighted average of all users
            hyrbid: nobias for topk neighbours 
            baseline: predicitons is mean of all users
        k: int, the number of nearest neighbours
        
        """        
        
        if cold_start:
            assert similarity_type=="attribute"\
                ,"Only similarity_type(attribute) support cold start,current:{}".format(similarity_type)
        
        self.similarity_type = similarity_type
        self.rate_similarity_metric = rate_similarity_metric
        self.rate_similarity_max_from_train = rate_similarity_max_from_train
        self.rate_similarity_scale = rate_similarity_scale
        self.attr_similarity_metric = attr_similarity_metric
        self.attr_similarity_max_from_train = attr_similarity_max_from_train
        self.attr_similarity_scale = attr_similarity_scale
        self.similarity_weight = similarity_weight
        self.use_bias = use_bias
        self.bias_type = bias_type
        self.predict_method = predict_method
        self.k = k
        self.n_jobs = n_jobs
        self.cold_start = cold_start
        self.verbose = verbose
        
    def train(self,x,y):
        
        _check_x(x,self.similarity_type)
        
        df = pd.DataFrame({"user":x["user"],"item":x["item"],"rate":y})
        df.loc[df['rate']==0, 'rate'] = EPSILON
        
        df_rate = df.pivot(index='user',columns='item',values='rate')
        df_rate = df_rate.fillna(0)
        array_rate = df_rate.values  
        
        if self.similarity_type in ["attribute","hybrid"]:
            array_attrs = x["attribute"]
        else:
            array_attrs = None
            
        if self.cold_start:
            self.model = _ColdStartModel(array_rate,array_attrs)
            self.model.fit(
                x = x,
                attr_similarity_metric = self.attr_similarity_metric,
                attr_similarity_max_from_train = self.attr_similarity_max_from_train,
                attr_similarity_scale = self.attr_similarity_scale, 
                num_threads = self.n_jobs
            )
            
        else:
            self.model = _Model(array_rate,array_attrs)
            self.model.fit(
                x,
                similarity_type = self.similarity_type,
                rate_similarity_metric = self.rate_similarity_metric,
                rate_similarity_max_from_train = self.rate_similarity_max_from_train,
                rate_similarity_scale = self.rate_similarity_scale,

                attr_similarity_metric = self.attr_similarity_metric,
                attr_similarity_max_from_train = self.attr_similarity_max_from_train,
                attr_similarity_scale = self.attr_similarity_scale,

                similarity_weight = self.similarity_weight,

                use_bias = self.use_bias,
                bias_type = self.bias_type,
                num_threads = self.n_jobs,
            )
            
    def predict(self,x):
        
        predictions = self.model.predict(
            x, predict_method=self.predict_method,k=self.k)
        
        return np.squeeze(np.array(predictions))
