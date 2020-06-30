import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from keras.models import Sequential, load_model
# from keras.layers import Dense,LSTM,Dropout,Conv1D,Flatten,MaxPooling1D
# from keras.callbacks import ModelCheckpoint
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold,RandomizedSearchCV,cross_val_score,cross_validate,train_test_split,KFold
from numpy.random import randint
import xgboost as xgb
from xgboost import XGBRFRegressor
from sklearn.metrics import r2_score
# m_check = ModelCheckpoint("./model/comp3/--{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)

def kaeri_metric(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''
    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)


### E1과 E2는 아래에 정의됨 ###

def E1(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)


def E2(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    
    
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))



train_feat = pd.read_csv('./data/dacon/comp3/train_features.csv',sep=',',header=0)
train_target = pd.read_csv('./data/dacon/comp3/train_target.csv',sep=',',header=0)
test_feat= pd.read_csv('./data/dacon/comp3/test_features.csv',sep=',',header=0)
submit = pd.read_csv('./data/dacon/comp3/sample_submission.csv',sep=',',header=0)

