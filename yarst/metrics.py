from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import scipy as sp
import pandas as pd
import sklearn.metrics as sk_metrics
from .env_setup import EPSILON


#-----------------------------------------
#       Regression Metrics
#-----------------------------------------

def _array_clean(y_true, y_pred, non_zero_contraint=False, ignore_nan = True):
    
    y_pred_array = np.array(y_pred).copy()
    y_true_array = np.array(y_true).copy()    
    
    if non_zero_contraint:
        y_pred_nonzero = y_pred_array[y_true_array.nonzero()].flatten()
        y_true_nonzero = y_true_array[y_true_array.nonzero()].flatten()
    else:
        y_pred_nonzero = y_pred.flatten()    
        y_true_nonzero = y_true.flatten()         
        
    if ignore_nan:
        nan_ix = np.isnan(y_pred_nonzero)
        y_pred_nan = y_pred_nonzero[~nan_ix]
        y_true_nan = y_true_nonzero[~nan_ix]
    else:
        y_pred_nan = y_pred_nonzero 
        y_true_nan = y_true_nonzero 
        
    return y_true_nan, y_pred_nan

def _bins(y_pred,bins=[-100,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) :
    _df_y_pred=pd.DataFrame(y_pred)
    _group_names = ['{:.2f}'.format(x) for x in bins[1:]]
    y_pred_label= np.array(pd.cut(_df_y_pred.iloc[:,0].values, bins, labels=_group_names))
  
    return y_pred_label

def _getrank(y_pred,ascending=True): 

    n=len(y_pred)
    temp = y_pred.argsort()
    y_pred_rank= np.empty(n, int)
    y_pred_rank[temp] = np.arange(n)
        
    if ascending==True :
        return y_pred_rank        
    else : 
        return np.array([n]*n)-y_pred_rank
    
def _cut_at_threshold(y_true,y_pred,threshold,return_part):
    
    assert return_part in ('high','low')
    
    if return_part== 'high':
        indices = np.argwhere(y_true>=threshold)
    else:
        indices = np.argwhere(y_true<threshold)
    return y_true[indices].flatten(),y_pred[indices].flatten()


def mse(non_zero_contraint=False, ignore_nan = True):
    def mean_squared_error(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)   
        return sk_metrics.mean_squared_error(_y_true, _y_pred)
    return mean_squared_error

def r2(non_zero_contraint=False, ignore_nan = True):
    def r2(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)   
        return sk_metrics.r2_score(_y_true, _y_pred)
    return r2

def rmse(non_zero_contraint=False, ignore_nan = True):
    def root_mean_squared_error(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)   
        return np.sqrt(sk_metrics.mean_squared_error(_y_true, _y_pred))
    return root_mean_squared_error 
        
def mae(non_zero_contraint=False, ignore_nan = True):
    def mean_absolute_error(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)   
        return sk_metrics.mean_absolute_error(_y_true, _y_pred)
    return mean_absolute_error         
        
def mape(non_zero_contraint=False, ignore_nan = True):
    def mean_absolute_percentage_error(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)   
        diff = np.absolute((_y_true - _y_pred)/ np.clip(np.absolute(_y_true), EPSILON, np.inf))
        return 100. * np.mean(diff)    
    return mean_absolute_percentage_error    

def msle(non_zero_contraint=False, ignore_nan = True):
    def mean_squared_logarithmic_error(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)  
        first_log = np.log(np.clip(_y_pred, EPSILON, np.inf) + 1.)
        second_log = np.log(np.clip(_y_true, EPSILON, np.inf) + 1.)
        return np.mean(np.square(first_log - second_log))   
    return mean_squared_logarithmic_error  

def kld(non_zero_contraint=False, ignore_nan = True):
    def kullback_leibler_divergence(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)  
        _y_true = np.clip(_y_true, EPSILON, 1)
        _y_pred = np.clip(_y_pred, EPSILON, 1)
        return np.sum(_y_true * np.log(_y_true / _y_pred))
    return kullback_leibler_divergence  

def crossentropy(non_zero_contraint=False, ignore_nan = True):
    def binary_crossentropy(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)  
        _y_pred= np.clip(_y_pred, EPSILON, 1.0-EPSILON)
        out = -(_y_true * np.log(_y_pred) + (1.0 - _y_true) * np.log(1.0 - _y_pred))
        return np.mean(out)  
    return binary_crossentropy

def mutual_info(bins=[-100,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                non_zero_contraint=False, ignore_nan = True):
    def mutual_info_fn(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)  
        _y_true_labels = _bins(_y_true,bins)
        _y_pred_labels = _bins(_y_pred,bins)
        return sk_metrics.mutual_info_score(_y_true_labels, _y_pred_labels)
    return mutual_info_fn

def normalized_mutual_info(bins=[-100,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                           non_zero_contraint=False, ignore_nan = True):
    def normalized_mutual_info_fn(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)  
        _y_true_labels = _bins(_y_true,bins)
        _y_pred_labels = _bins(_y_pred,bins)
        return sk_metrics.normalized_mutual_info_score(_y_true_labels, _y_pred_labels)
    return normalized_mutual_info_fn

def spear_corr(non_zero_contraint=False, ignore_nan = True):
    def spear_corr_fn(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)  
        return sp.stats.spearmanr(_y_true,_y_pred)[0]
    return spear_corr_fn

def pearson_corr(non_zero_contraint=False, ignore_nan = True):
    def pearson_corr_fn(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)  
        return sp.stats.pearsonr(_y_true,_y_pred)[0]
    return pearson_corr_fn

def kendall_corr(ascending=True,non_zero_contraint=False, ignore_nan = True):
    def kendall_corr_fn(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)  
        _y_true_rank=_getrank(_y_true,ascending)
        _y_pred_rank=_getrank(_y_pred,ascending)
        return sp.stats.kendalltau(_y_true_rank, _y_pred_rank)[0]	
    return kendall_corr_fn


def mse_h(threshold=0.5,non_zero_contraint=False, ignore_nan = True):
    def mean_squared_error_at_high(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan) 
        _y_true, _y_pred = _cut_at_threshold(_y_true, _y_pred,threshold,'high')
  
        return sk_metrics.mean_squared_error(_y_true, _y_pred)
    return mean_squared_error_at_high

def mse_l(threshold=0.5,non_zero_contraint=False, ignore_nan = True):
    def mean_squared_error_at_low(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan) 
        _y_true, _y_pred = _cut_at_threshold(_y_true, _y_pred,threshold,'low')
  
        return sk_metrics.mean_squared_error(_y_true, _y_pred)
    return mean_squared_error_at_low

def mae_h(threshold=0.5,non_zero_contraint=False, ignore_nan = True):
    def mean_absolute_error_at_high(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan) 
        _y_true, _y_pred = _cut_at_threshold(_y_true, _y_pred,threshold,'high')
  
        return sk_metrics.mean_absolute_error(_y_true, _y_pred)
    return mean_absolute_error_at_high

def mae_l(threshold=0.5,non_zero_contraint=False, ignore_nan = True):
    def mean_absolute_error_at_low(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan) 
        _y_true, _y_pred = _cut_at_threshold(_y_true, _y_pred,threshold,'low')
  
        return sk_metrics.mean_absolute_error(_y_true, _y_pred)
    return mean_absolute_error_at_low

def r2_h(threshold=0.5,non_zero_contraint=False, ignore_nan = True):
    def r2_at_high(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan) 
        _y_true, _y_pred = _cut_at_threshold(_y_true, _y_pred,threshold,'high')
  
        return sk_metrics.r2_score(_y_true, _y_pred)
    return r2_at_high

def r2_l(threshold=0.5,non_zero_contraint=False, ignore_nan = True):
    def r2_at_low(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan) 
        _y_true, _y_pred = _cut_at_threshold(_y_true, _y_pred,threshold,'low')
  
        return sk_metrics.r2_score(_y_true, _y_pred)
    return r2_at_low

def max_value(non_zero_contraint=False, ignore_nan = True):
    def maximum_value(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)         
        
        return np.amax(y_pred)
    return maximum_value
        
def min_value(non_zero_contraint=False, ignore_nan = True):
    def minimum_value(y_true, y_pred):
        _y_true, _y_pred = _array_clean(
            y_true, y_pred, 
            non_zero_contraint = non_zero_contraint, 
            ignore_nan = ignore_nan)         
        
        return np.amin(y_pred)
    return minimum_value        
        
        
#-----------------------------------------
#       Binary Classification Metrics
#-----------------------------------------

def _binary_prob2label(y_pred,threshold=0.5,out_type=np.int):
    
    _y_pred = y_pred.copy()
    
    _y_pred[_y_pred>threshold] = 1
    _y_pred[_y_pred<=threshold] = 0

    return _y_pred.astype(out_type)

def average_precision(y_true, y_pred):
    return sk_metrics.average_precision_score(y_true, y_pred)

def roc_auc(y_true, y_pred):
    return sk_metrics.roc_auc_score(y_true, y_pred)

def accuracy_at_threshold(threshold=0.5):
    def accuracy(y_true, y_pred):
        _y_pred = _binary_prob2label(y_pred,threshold)
        return sk_metrics.accuracy_score(y_true, _y_pred)
    return accuracy

def precision_at_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        _y_pred = _binary_prob2label(y_pred,threshold)
        return sk_metrics.precision_score(y_true, _y_pred)          
    return precision

def recall_at_threshold(threshold=0.5):
    def recall(y_true, y_pred):
        _y_pred = _binary_prob2label(y_pred,threshold)
        return sk_metrics.recall_score(y_true, _y_pred)       
    return recall

def f1_at_threshold(threshold=0.5):
    def f1(y_true, y_pred):
        _y_pred = _binary_prob2label(y_pred,threshold)
        return sk_metrics.f1_score(y_true, _y_pred)        
    return f1

def precision_at_topk(topk_percent=0.1,threshold=0.5):
    def topk_precision(y_true, y_pred):
        y_pred_sort=np.sort(y_pred)[::-1]
        total=y_pred_sort.shape[0]
        k= int(total * topk_percent)
        topk_threshold=y_pred_sort[k-1]
        _y_pred = _binary_prob2label(y_pred,topk_threshold)
        return sk_metrics.precision_score(y_true, _y_pred)        
    return topk_precision

def recall_at_topk(topk_percent=0.1,threshold=0.5):
    def topk_recall(y_true, y_pred):
        y_pred_sort=np.sort(y_pred)[::-1]
        total=y_pred_sort.shape[0]
        k= int(total * topk_percent)
        topk_threshold=y_pred_sort[k-1]
        _y_pred = _binary_prob2label(y_pred,topk_threshold)
        return sk_metrics.recall_score(y_true, _y_pred)        
    return topk_recall

def f1_at_topk(topk_percent=0.1,threshold=0.5):
    def topk_f1(y_true, y_pred):
        y_pred_sort=np.sort(y_pred)[::-1]
        total=y_pred_sort.shape[0]
        k= int(total * topk_percent)
        topk_threshold=y_pred_sort[k-1]
        _y_pred = _binary_prob2label(y_pred,topk_threshold)
        return sk_metrics.f1_score(y_true, _y_pred)        
    return topk_f1

