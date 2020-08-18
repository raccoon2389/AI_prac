import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout,Flatten, Conv2DTranspose
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import RMSprop

m_check =  ModelCheckpoint(filepath=".\model\comp6--{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)
def schedul(epoch,lr):
    if epoch < 7:
        print(lr)
        return lr
    else:
        return lr * np.exp(-0.1)
sched = LearningRateScheduler(schedul)
train = pd.read_csv('./Data/dacon/comp6/train.csv', header=0, index_col=0)
test = pd.read_csv('./Data/dacon/comp6/test.csv', header=0, index_col=0)
sub = pd.read_csv('./Data/dacon/comp6/submission.csv', header=0, index_col=0)

# print(train.isnull().sum().sum()) #null없음

train_y = train.loc[:,'digit'].values.reshape(-1)
train_x = train.loc[:, '0':].values.reshape(-1, 28, 28,1)/255.

# test_y = test.loc[:,'digit'].values.reshape(-1, 2)
test_x = test.loc[:,'0':].values.reshape(-1, 28, 28, 1)/255.
# print(train_x[1])
# plt.imshow(train_x[1].reshape(28,28))
# plt.show()
# imabe = Image.fromarray(train_x[0])
# imabe.show()

train_y = to_categorical(train_y)
# print(train_y)

# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)


def autoencoder(hidden_laysey_size,model):
    model.add(Conv2D(hidden_laysey_size, (2, 2),
                     padding='valid', input_shape=(28, 28, 1)))
    model.add(Conv2D(hidden_laysey_size, (2, 2),
                     padding='valid'))
    model.add(Conv2D(hidden_laysey_size, (2, 2),
                     padding='valid'))
    model.add(Conv2D(hidden_laysey_size, (2, 2),
                     padding='valid'))
    model.add(Conv2DTranspose(1, (5, 5), padding='valid'))

    return model


def create_model(nodes,model):
    model.add(Conv2D(nodes*2, (2, 2), strides=1,
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(nodes*2, (2, 2), strides=1,
                     activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(nodes*4, (2, 2),padding='valid', strides=1,
                     activation='relu'))
    model.add(Conv2D(nodes*4, (2, 2), padding='valid', strides=1,
                     activation='relu'))
    model.add(Conv2D(nodes*4, (2, 2), padding='valid', strides=1,
                     activation='relu'))
    model.add(Conv2D(nodes*4, (2, 2), padding='valid', strides=1,
                     activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(nodes*8, (2, 2), padding='valid', strides=1,
                     activation='relu'))
    model.add(Conv2D(nodes*8, (2, 2), padding='valid', strides=1,
                     activation='relu'))
    model.add(Conv2D(nodes*8, (2, 2), padding='valid', strides=1,
                     activation='relu'))
    # model.add(Conv2D(nodes*8, (2, 2), padding='valid', strides=1,
    #                  activation='relu'))
    # model.add(Conv2D(nodes*8, (2, 2), padding='valid', strides=1,
    #                  activation='relu'))
    # model.add(Conv2D(nodes*8, (2, 2), padding='valid', strides=1,
    #                  activation='relu'))
    # model.add(MaxPool2D(2,2))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.summary()
    optimizer = RMSprop(lr = 0.001, epsilon = 1e-8)
    model.compile(optimizer= optimizer, loss= 'categorical_crossentropy',metrics=['acc'])
    return model

def train():
    model = Sequential()
    model = autoencoder(32,model)
    model = create_model(32,model)
    model.fit(x=train_x, y=train_y, batch_size=30, epochs=100,
              validation_split=0.25, callbacks=[m_check])
    # pred = model.predict(train_x,batch_size=30)
    # print(pred)

def pred():
    # print(sub)
    model = load_model('./model/comp6--11--0.6531.hdf5')

    pred = model.predict(test_x,batch_size=30)
    pred = np.argmax(pred,axis=1)
    pred_df = pd.DataFrame(pred,index=range(2049,22529),columns=["digit"])
    pred_df.index.name = "id"
    pred_df.to_csv("./comp6.csv",index=True)
    print('Done')

train()
# pred()
