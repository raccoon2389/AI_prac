import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout,Flatten, Conv2DTranspose

train = pd.read_csv('./Data/dacon/comp6/train.csv', header=0, index_col=0)
test = pd.read_csv('./Data/dacon/comp6/test.csv', header=0, index_col=0)
sub = pd.read_csv('./Data/dacon/comp6/submission.csv', header=0, index_col=0)

print(train.isnull().sum().sum()) #null없음

train_y = train.loc[:,'digit'].values.reshape(-1)
train_x = train.loc[:, '0':].values.reshape(-1, 28, 28, 1)

test_y = test.loc[:,:'letter'].values.reshape(-1, 2)
test_x = test.loc[:,'0':].values.reshape(-1, 28, 28, 1)

train_y = to_categorical(train_y)
print(train_y.shape)

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
    model.add(Conv2D(nodes*4, (2, 2),padding='same', strides=1,
                     activation='relu'))
    model.add(Conv2D(nodes*4, (2, 2), padding='same', strides=1,
                     activation='relu'))
    model.add(Conv2D(nodes*4, (2, 2), padding='same', strides=1,
                     activation='relu'))
    model.add(Conv2D(nodes*4, (2, 2), padding='same', strides=1,
                     activation='relu'))
    model.add(MaxPool2D((2,2)))
    # model.add(Conv2D(nodes*8, (2, 2), padding='same', strides=1,
    #                  activation='relu', input_shape=(28, 28, 1)))
    # model.add(Conv2D(nodes*8, (2, 2), padding='same', strides=1,
    #                  activation='relu', input_shape=(28, 28, 1)))
    # model.add(Conv2D(nodes*8, (2, 2), padding='same', strides=1,
    #                  activation='relu', input_shape=(28, 28, 1)))
    # model.add(Conv2D(nodes*8, (2, 2), padding='same', strides=1,
    #                  activation='relu', input_shape=(28, 28, 1)))
    # model.add(Conv2D(nodes*8, (2, 2), padding='same', strides=1,
    #                  activation='relu', input_shape=(28, 28, 1)))
    # model.add(Conv2D(nodes*8, (2, 2), padding='same', strides=1,
    #                  activation='relu', input_shape=(28, 28, 1)))
    # model.add(MaxPool2D(2,2))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(256))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(10))
    model.summary()
    model.compile(optimizer= 'adam', loss= 'categorical_crossentropy',metrics=['acc'])
    return model

def train():
    model = Sequential()
    model = autoencoder(32,model)
    model = create_model(32,model)
    model.fit(train_x,train_y,epochs=30,validation_split=0.25)

train()
