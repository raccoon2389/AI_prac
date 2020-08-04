from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
# Dropout 중간에 노드들을 일정비율을 지운다.
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import random


def autoencoder(hidden_laysey_size):
    model = Sequential()
    model.add(Dense(units=hidden_laysey_size,
                    input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model


(x_train, y_train), (x_test, y_test) = mnist.load_data()

#데이터 전처리 1. 원핫인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

#데이터 전처리 2. 정규화
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1]*x_train.shape[2]).astype('float32')/255.
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1]*x_test.shape[2]).astype('float32')/255.


model01 = autoencoder(1)
model02 = autoencoder(2)
model03 = autoencoder(4)
model04 = autoencoder(8)
model05 = autoencoder(16)
model06 = autoencoder(32)

model01.compile(optimizer='adam',loss = 'mse',metrics=['acc'])
model02.compile(optimizer='adam',loss = 'mse',metrics=['acc'])
model03.compile(optimizer='adam',loss = 'mse',metrics=['acc'])
model04.compile(optimizer='adam',loss = 'mse',metrics=['acc'])
model05.compile(optimizer='adam',loss = 'mse',metrics=['acc'])
model06.compile(optimizer='adam',loss = 'mse',metrics=['acc'])
model01.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model02.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model03.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model04.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model05.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model06.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model01.fit(x_train,x_train,epochs=10)
model02.fit(x_train,x_train,epochs=10)
model03.fit(x_train,x_train,epochs=10)
model04.fit(x_train,x_train,epochs=10)
model05.fit(x_train,x_train,epochs=10)
model06.fit(x_train,x_train,epochs=10)

output01=model01.predict(x_test)
output02=model02.predict(x_test)
output03=model03.predict(x_test)
output04=model04.predict(x_test)
output05=model05.predict(x_test)
output06=model06.predict(x_test)

fig, axes = plt.subplots(7, 5, figsize=(20, 7))
outputs = [x_test,
           output01,
           output02,
           output03,
           output04,
           output05,
           output06]

random_images = random.sample(range(output01.shape[0]), 5)

for row_num,row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()