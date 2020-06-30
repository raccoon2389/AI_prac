import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense,LSTM,Dropout,Conv2D,Flatten,MaxPool2D,BatchNormalization,Lambda
from keras.callbacks import ModelCheckpoint
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold,RandomizedSearchCV,cross_val_score,cross_validate,train_test_split,KFold
from numpy.random import randint
# import xgboost as xgb
# from xgboost import XGBRFRegressor
# from sklearn.metrics import r2_score
import keras.backend as K
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
    
    # _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    
    return K.mean(K.square(y_true-y_pred))/2e+04


def E2(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)])
    return K.mean(K.square(divResult))



train_feat = pd.read_csv('./data/dacon/comp3/train_features.csv',sep=',',header=0)
train_target = pd.read_csv('./data/dacon/comp3/train_target.csv',sep=',',header=0)
test_feat= pd.read_csv('./data/dacon/comp3/test_features.csv',sep=',',header=0)
submit = pd.read_csv('./comp3.csv',sep=',',header=0)
XY = pd.read_csv('./data/dacon/comp3/XY2.csv',index_col=0,header=0)


x = np.load('./data/dacon/comp3/feat.npy')
y = np.load('./data/dacon/comp3/target.npy')
test = np.load('./data/dacon/comp3/test.npy')

x = x.reshape(2800,375,4,1)
y = y[:,2:]
test = test.reshape(700,375,4,1)

weight2 = np.array([1,1])
def set_model(train_target):  # 0:x,y, 1:m, 2:v

    
    activation = 'elu'
    padding = 'valid'
    model = Sequential()
    nf = 64
    fs = (3,1)

    model.add(Conv2D(nf,fs, padding=padding, activation=activation,input_shape=(375,4,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*2,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*4,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*8,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*16,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*32,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 1)))

    model.add(Flatten())
    model.add(Dense(128, activation ='elu'))
    model.add(Dense(64, activation ='elu'))
    model.add(Dense(32, activation ='elu'))
    model.add(Dense(16, activation ='elu'))
    model.add(Dense(1))


    model.compile(optimizer='adam',loss=E2,metrics=['mse'])


    global weight2
    if train_target == 1: # only for M
        weight2 = np.array([1,0])
    else: # only for V
        weight2 = np.array([0,1])
    model.summary()

    return model

for i in range(2):
    model = set_model(i)
    if i == 0:
        print(1)
        model = load_model('./model/comp3/M--93--0.0005.hdf5',custom_objects={'E2':E2})
        # m_check = ModelCheckpoint("./model/comp3/M--{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)

        # model.fit(x,y[:,0],batch_size=10,epochs=100,validation_split=0.2,callbacks=[m_check])
        pred = model.predict(test)
        df = pd.DataFrame(pred , index=list(range(2800,3500)),columns=["M"])
        df.index.name = "id"
        # print(pred)

    else:
        print(1)

        model2 = load_model('./model/comp3/V--66--0.0032.hdf5',custom_objects={'E2':E2})
        # m_check = ModelCheckpoint("./model/comp3/V--{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)

        # model.fit(x,y[:,1],batch_size=10,epochs=100,validation_split=0.2,callbacks=[m_check])
        pred = model2.predict(test)
        df["V"] = pred
        # print(pred)

# df.to_csv('./data/dacon/comp3/MV3.csv')
print(df)

sub = pd.concat([XY,df],axis=1)
sub.to_csv("./COMP3MV.csv")
print(sub.head())

