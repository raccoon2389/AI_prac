import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Dense,LSTM, Conv2D,Flatten,MaxPooling2D,Dropout, Input
from keras.callbacks import EarlyStopping

e_stop = EarlyStopping(monitor='loss',patience=5,mode='auto')
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0],28,28)/255.
x_test = x_test.reshape(x_test.shape[0],28,28)/255.

print(x_train.shape)

input1 = Input(shape=(x_train.shape[1],x_train.shape[2]))

hid = LSTM(10,activation='relu')(input1)
hid = Dropout(0.2)(hid)
hid = Dense(50,activation='relu')(hid)
hid = Dropout(0.2)(hid)
output1 = Dense(10,activation='softmax')(hid)

model = Model(inputs=[input1], outputs=[output1])

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train,y_train,batch_size=200,epochs=100,validation_split=0.3, callbacks=[e_stop])

loss, acc = model.evaluate(x_test,y_test,batch_size=100)
print(loss,acc)
# loss : 0.4732126519083977 acc : 0.8270999789237976