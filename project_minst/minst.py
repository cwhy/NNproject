# coding: utf-8
import pandas as pd
import numpy as np


# In[11]:

train_raw = pd.read_csv('./data/train.csv')
test_raw = pd.read_csv('./data/test.csv')

def dat_get(datset):
    labels = dataset[:,-1]
    features = dataset[:,0:-1]
    return features, labels


def compare(lb1, lb2):
    np.sum(lb1 == lb2)
    return features, labels


# In[14]:

train_all = train_raw.values
to_train = train_all[0:1000,:]
to_validate = train_all[1001:2000,:]
tr_ft, tr_lb = dat_get(to_train)
va_ft, va_lb = dat_get(to_validate)


# In[21]:

tr_ft.shape

