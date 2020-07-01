import os
import json
import numpy as np
# from tqdm import tqdm
# import jovian
# import kaeri_metric
# from kaeri_metric import E1, E2, kaeri_metric, E2M, E2V
# %matplotlib inline
# import matplotlib as plt
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten,MaxPooling2D,BatchNormalization,Lambda, AveragePooling2D,Dropout
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from numpy.random import randint
from sklearn.model_selection import RandomizedSearchCV
import glob
# X_data = []
# Y_data = []

X_data = np.loadtxt('./data/dacon/comp3/train_features.csv',skiprows=1,delimiter=',')
X_data = X_data[:,1:]
print(X_data.shape)
     
    
Y_data = np.loadtxt('./data/dacon/comp3/train_target.csv',skiprows=1,delimiter=',')
Y_data = Y_data[:,1:]
print(Y_data.shape)

X_data = X_data.reshape((2800,375,5,1))
print(X_data.shape)

X_data_test = np.loadtxt('./data/dacon/comp3/test_features.csv',skiprows=1,delimiter=',')
X_data_test = X_data_test[:,1:]
X_data_test = X_data_test.reshape((700,375,5,1))

from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2)
# X_train = X_data
# Y_train = Y_data
print(X_train.shape)

weight1 = np.array([1,1,0,0])
weight2 = np.array([0,0,1,1])

def my_loss(y_true, y_pred):
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)])
    return K.mean(K.square(divResult))


def my_loss_E1(y_true, y_pred):
    return K.mean(K.square(y_true-y_pred)*weight1)/2e+04

def my_loss_E2(y_true, y_pred):
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)])
    return K.mean(K.square(divResult)*weight2)


# tr_target = 2 

def set_model(train_target,drop=0.3,nf=64,f=3):  # 0:x,y, 1:m, 2:v
    fs = (f,1)
    fn = (f,4)
    activation = 'elu'
    padding = 'valid'
    model = Sequential()

    model.add(Conv2D(nf,fs, padding=padding, activation=activation,input_shape=(375,5,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(drop))

    model.add(Conv2D(nf*2,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(drop))

    model.add(Conv2D(nf*4,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(drop))

    model.add(Conv2D(nf*8,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(drop))


    model.add(Conv2D(nf*16,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(drop))

    model.add(Conv2D(nf*32,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(drop))

    model.add(Flatten())
    model.add(Dense(512, activation ='elu'))
    model.add(Dropout(drop))

    model.add(Dense(256, activation ='elu'))
    model.add(Dropout(drop))
    model.add(Dense(128, activation ='elu'))
    model.add(Dropout(drop))
    model.add(Dense(64, activation ='elu'))
    model.add(Dropout(drop))
    model.add(Dense(32, activation ='elu'))
    model.add(Dropout(drop))
    model.add(Dense(4))

    optimizer = keras.optimizers.Adam()

    global weight2
    if train_target == 1: # only for M
        weight2 = np.array([0,0,1,0])
    else: # only for V
        weight2 = np.array([0,0,0,1])
       
    if train_target==0:
        model.compile(loss=my_loss_E1,
                  optimizer=optimizer,metrics=['acc']
                 )
    else:
        model.compile(loss=my_loss_E2,
                  optimizer=optimizer,metrics=['acc']
                 )
       
    model.summary()

    return model


# tr_target = 2 

def train(train_target,X,Y):

    MODEL_SAVE_FOLDER_PATH = f"./model/{train_target}/"
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)
    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
    best_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')

    modelK = KerasRegressor(build_fn=set_model,verbose=1)
    param = create_hyper(train_target)
    search = RandomizedSearchCV(estimator=modelK,param_distributions=param,n_iter=30,cv=None,n_jobs=1)

    search.fit(X, Y,
                  epochs=100,
                  shuffle=True,
                  validation_split=0.2,
                  verbose = 2,
                  callbacks=[best_save])
    print(search.best_estimator_)
    return search.estimator


def load_best_model(train_target):
    PATH = f"./model/{train_target}/"
    f_list = glob.glob(PATH+"*")
    f_list = f_list.split('-')
    L = len(f_list)
    for i in list(range(0,L,2)):
        f_list2 = f_list(i)
    b = np.argmin(f_list2)
    best = f_list[b] 

    model_path = f"./model/{train_target}/{best}"
    if train_target == 0:
        model = load_model('best_m.hdf5' , custom_objects={'my_loss_E1': my_loss, })
    else:
        model = load_model('best_m.hdf5' , custom_objects={'my_loss_E2': my_loss, })

    score = model.evaluate(X_data, Y_data, verbose=0)
    print('loss:', score)

    pred = model.predict(X_data)

    i=0

    print('정답(original):', Y_data[i])
    print('예측값(original):', pred[i])

    return model

def create_hyper(train_target):
    batches = list(range(10,110,10))
    dropout = np.linspace(0.1,0.5,20).tolist()
    nf = randint(8,128,30).tolist()
    fs = list(range(2,6))
    return{"batch_size" : batches, "drop" : dropout, "nf" : nf, "f" : fs,"train_target" : [train_target,train_target]}


submit = pd.read_csv('./data/dacon/comp3/sample_submission.csv')


for train_target in range(3):
    train(train_target,X_train, Y_train)    


for train_target in range(3):
    best_model = load_best_model(train_target)

   
    pred_data_test = best_model.predict(X_data_test)
    
    
    if train_target == 0: # x,y 학습
        submit.iloc[:,1] = pred_data_test[:,0]
        submit.iloc[:,2] = pred_data_test[:,1]

    elif train_target == 1: # m 학습
        submit.iloc[:,3] = pred_data_test[:,2]

    elif train_target == 2: # v 학습
        submit.iloc[:,4] = pred_data_test[:,3]

submit.to_csv('./base.csv')