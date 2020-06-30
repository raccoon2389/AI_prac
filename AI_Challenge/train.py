import numpy as np
import pandas as pd
from keras.models import Sequential,load_model
from keras.layers import Dense, Conv2D,MaxPool2D,Dropout,Flatten
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from keras.utils.io_utils import HDF5Matrix
import os 

import h5py
import numpy as np
import os
import pandas as pd
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))+"/d"
list_train = path+'/train/train_labels.txt'
list_val = path +'/validate/validate_labels.txt'
list_test = path+'/test/test_labels.txt'
list_val = pd.read_csv(list_val,sep=' ',header=None)
list_train = pd.read_csv(list_train,sep=' ',header=None)
list_test = pd.read_csv(list_test,sep=' ',header=None)
# print(list_val)


print(list_test)

'''
with h5py.File('/tf/notebooks/pre.hdf5','w') as f:
    f.create_dataset('x_train',(38013,150,150,3),dtype='float32')
    f.create_dataset('y_train_P',(38013,1),dtype='float32')
    f.create_dataset('y_train_D',(38013,1),dtype='float32')
    f.create_dataset('x_val',(8146,150,150,3),dtype='float32')
    f.create_dataset('y_val_P',(8146,1),dtype='float32')
    f.create_dataset('y_val_D',(8146,1),dtype='float32')
    f.create_dataset('x_test',(8147,150,150,3),dtype='float32')
    f.create_dataset("y_test_P",(8147,1),dtype='float32')
    f.create_dataset("y_test_D",(8147,1),dtype='float32')

    x_train = f["x_train"]
    y_train_P = f["y_train_P"]
    x_val = f["x_val"]
    y_val_P = f["y_val_P"]
    x_test = f["x_test"]
    y_test_P = f["y_test_P"]
    y_test_ = f["y_test_P"]


    # dat = Image.open(f"{path}/train/{list_train.iloc[0,0]}").resize(((150,150)))

    # x_train[0] = np.asarray(dat)/255.
    # print(list_train.iloc[i,1:].values.shape)
    # y_train[0] = list_train.iloc[0,1:].values/1.0

    for i in range(38012):
        dat = Image.open(f"{path}/train/{list_train.iloc[i,0]}").resize(((150,150)))
        x = np.asarray(dat)/255.
        if x.shape[2]==4:
            x = x[:,:,:3]
        x_train[i] = x
        # print(x_train[i].shape)
        # print(list_train.iloc[i,1:].values.shape)
        # v = list_train.iloc[i,1:].values
        # v = v[:]
        print(i)
        y_train_P[i] = list_train.iloc[i,1].astype('float32')
        f['y_train_D'][i] = list_train.iloc[i,2].astype('float32')

    for i in range(8146):
        print(i)
        dat = Image.open(f"{path}/test/{list_test.iloc[i,0]}").resize(((150,150)))
        x = np.asarray(dat)/255.
        if x.shape[2]==4:
            x = x[:,:,:4]
        x_test[i] = x
    for i in range(8145):
        print(i)

        dat = Image.open(f"{path}/validate/{list_val.iloc[i,0]}").resize(((150,150)))
        x = np.asarray(dat)/255.
        if x.shape[2]==4:
            x = x[:,:,:4]
        x_val[i] = x 
        y_val_P[i] =list_val.iloc[i,1].astype('float32')
        f['y_val_D'][i] =list_val.iloc[i,2].astype('float32')
'''

path = os.path.dirname(os.path.abspath(__file__))+"/d"
list_test = path+'/test/test_labels.txt'
list_test = pd.read_csv(list_test,sep=' ',header=None,index_col=None)



x_train= HDF5Matrix('/tf/notebooks/d/pre.hdf5','x_train')
y_train_P= HDF5Matrix('/tf/notebooks/d/pre.hdf5','y_train_P')
y_train_D= HDF5Matrix('/tf/notebooks/d/pre.hdf5','y_train_D')
x_val= HDF5Matrix('/tf/notebooks/d/pre.hdf5','x_val')
y_val_P= HDF5Matrix('/tf/notebooks/d/pre.hdf5','y_val_P')
y_val_D= HDF5Matrix('/tf/notebooks/d/pre.hdf5','y_val_D')
x_test= HDF5Matrix('/tf/notebooks/d/pre.hdf5','x_test')

y_train_P = np_utils.to_categorical(y_train_P)
y_train_D = np_utils.to_categorical(y_train_D)
y_val_P = np_utils.to_categorical(y_val_P)
y_val_D = np_utils.to_categorical(y_val_D)

print(y_train_D.shape)
print(y_train_P.shape)
print(y_val_D.shape)
print(y_val_P.shape)

print(y_val_D)


def create_model(idx):
    model = Sequential()
    model.add(Conv2D(200,(3,3),activation='elu',input_shape=(150,150,3)))
    model.add(Dropout(0.3))
    model.add(MaxPool2D(2,2))
    # model.add(Conv2D(200,(3,3),activation='elu'))
    # model.add(Dropout(0.3))
    # model.add(MaxPool2D(2,2))
    # model.add(Conv2D(100,(3,3),padding='same',activation='elu'))
    # model.add(Dropout(0.3))
    # model.add(MaxPool2D(2,2))
    model.add(Conv2D(100,(3,3),padding='same',activation='elu'))
    model.add(Dropout(0.3))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(100,(2,2),padding='same',activation='elu'))
    model.add(Dropout(0.3))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(100,(2,2),padding='same',activation='elu'))
    model.add(Dropout(0.3))
    model.add(MaxPool2D(2,2))

    model.add(Flatten())
    model.add(Dense(500,activation="elu"))
    model.add(Dropout(0.3))
    model.add(Dense(100,activation="elu"))
    model.add(Dropout(0.3))
    # model.add(Dense(100,activation="elu"))
    # model.add(Dropout(0.3))
    model.add(Dense(100,activation="elu"))
    model.add(Dropout(0.3))

    model.add(Dense(100,activation="elu"))
    model.add(Dropout(0.3))

    model.add(Dense(idx,activation="softmax"))

    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['acc'])
    model.summary()
    return model

seed = np.random.seed(7)
models = []
kf = KFold(n_splits=3, shuffle=True,random_state=seed)
'''
for i in range(2):
    labels = [14,21]
    print(i)
    models.append(create_model(labels[i]))
for i in range(1,2):
    print(i)
    y = [y_train_P,y_train_D]
    y_v = [y_val_P,y_val_D]
    print(y_train_P.shape)
    if i ==0:
        m_check = ModelCheckpoint("/tf/notebooks/model/P--{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)
    else:
        m_check = ModelCheckpoint("/tf/notebooks/model/D--{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)

    models[i].fit(x_train,y[i],batch_size=100,epochs=50,callbacks=[m_check],validation_data=(x_val,y_v[i]),shuffle="batch")
    pred = models[i].predict(x_test)
    df = pd.DataFrame(pred)
    df.to_csv(f"/tf/notebooks/{i}.csv")
'''
# for train_i,test_i in kf.split(x):
#     train_x,train_y = x[train_i],y[train_i]
#     test_x, test_y = x[test_i], y[test_i]

#     model.fit(train_x,train_y,batch_size=30,epochs=100,validation_split=0.25,callbacks=[m_check])
#     score = model.evaluate(test_x,test_y)
#     print(score)

# 

for i in range(2):
    idx=0
    pred = np.zeros((8147,))
    if i == 0:
        model = load_model('/tf/notebooks/model/P--22--1.8938.hdf5')
        predy = model.predict(x_test)
        pred = np.argmax(predy,axis=1) 

        # for p in predy:
        #     pred[idx] = np.argmax(p).astype('int32')
        #     idx+=1
        print(pred.shape)
        

        list_test.iloc[:,1] = pred
    else:
        model = load_model('/tf/notebooks/model/D--24--1.8480.hdf5')
        predy = model.predict(x_test)
        pred = np.argmax(predy,axis=1) 

        # for p in predy:
        #     pred[idx] = np.argmax(p).astype('int32')
        #     idx+=1
        print(pred.shape)

        list_test.iloc[:,2] = pred

list_test.to_csv('/tf/notebooks/out/output.txt',index=False,columns=None)

