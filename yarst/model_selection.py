from __future__ import division
from __future__ import print_function 

import numpy as np
import pandas as pd

def TrainTestSplit(x,x_col,split_count = 1,threshold = 2, fraction = None,verbose=0):
    """
    Split ratings data into train and test sets
    Params
    ------
    x : pandas data frame, 
        three columns type
    x_col : dict, define the col name 
    split_count: int
        Number of user-item-ratings per user to move from training to test set.
    threshold: int
        set the minimin number of items that test has hold
    fractions : float
        Fraction of users to split off some of their interactions into test set. 
        If None, then all users are considered.
    verbose: bool
        print information
    ========
    return:
    train_combined: (pd.DataFrame)
        train datasets, long table
    test_long: (pd.DataFrame)
        test datasets, long table
              
    """

    assert isinstance(x, pd.DataFrame)
    assert split_count > 0
    assert threshold > split_count 
    assert x_col['user'] is not None,'please set user in x_col'
    assert x_col['item'] is not None,'please set item in x_col'
    assert x_col['rate'] is not None,'please set rate in x_col'
    
    user_col = x_col['user']
    item_col = x_col['item']
    rate_col = x_col['rate']
    
    x_sorted = x[[user_col, item_col, rate_col]].sort_values([user_col, item_col],ascending=[True, True]).reset_index(drop=True)    
    stat = x_sorted.groupby(user_col)[item_col].transform('count').reset_index().copy()
    data_for_split = x_sorted[stat[item_col] >= threshold ].copy()
    data_remaining = x_sorted[stat[item_col] < threshold ].copy()
    
    # convert to rating table
    data_for_split.loc[data_for_split[rate_col] == 0, rate_col] = 1e-07
    rating = data_for_split.pivot(index=user_col,columns=item_col,values=rate_col).values
    rating[np.isnan(rating)] = 0

    test = np.zeros(rating.shape)
    train = rating.copy()

    if fraction:
        try:
            user_index = np.random.choice(
                np.where((pd.DataFrame(train) != 0).sum(axis=1).values >= split_count + 1)[0],
                replace=False,
                size=np.int64(np.floor(fraction * train.shape[0]))
            ).tolist()
        except:
            print(('Not enough users with > {} '
                   'interactions for fraction of {}') \
                  .format(2 * split_count, fraction))
            raise
    else:
        user_index = range(train.shape[0])
    
    for user in user_index:
        try:
            test_rating = np.random.choice(rating[user, :].nonzero()[0],
                                            size=split_count,
                                            replace=False)
            train[user, test_rating] = 0.
            test[user, test_rating] = rating[user, test_rating]
        except:
            print ('some user have items less than {}'.format(2 * split_count))
            raise

    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))

    # convert wide to long
    train = pd.DataFrame(train)
    train['index'] = train.index
    train_long = pd.melt(train,id_vars = ['index'])
    train_long.columns = [user_col,item_col,rate_col]

    train_long = train_long[train_long[rate_col] != 0]
    train_long.loc[train_long[rate_col] == 1e-07,rate_col]= 0
    
    train_combined = pd.concat([train_long,data_remaining],axis=0).reset_index(drop=True)

    test = pd.DataFrame(test)
    test['index'] = test.index
    test_long = pd.melt(test,id_vars = ['index'])
    test_long.columns = [user_col,item_col,rate_col]
    
    test_long = test_long[test_long[rate_col] != 0]
    test_long.loc[test_long[rate_col] == 1e-07,rate_col ] = 0
    test_long = test_long.reset_index(drop=True)  
    
    if verbose:

        train_sparsity = float(train_combined.shape[0])
        train_sparsity /= (train_combined[user_col].unique().shape[0] * train_combined[item_col].unique().shape[0])
        train_sparsity *= 100
                           
        test_sparsity = float(test_long.shape[0])
        test_sparsity /= (test_long[user_col].unique().shape[0] * test_long[item_col].unique().shape[0])
        test_sparsity *= 100
                           
        print ('{} unique users in train'.format(train_combined[user_col].unique().shape[0]))
        print ('Train sparsity: {:4.2f}%'.format(train_sparsity))
        print ('{} unique users in test'.format(test_long[user_col].unique().shape[0]))  
        print ('Test sparsity: {:4.2f}%'.format(test_sparsity))
                      
                           
    return train_combined, test_long
