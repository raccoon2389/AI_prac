import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Dense,LSTM, Conv2D,Flatten,MaxPooling2D,Dropout, Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, TensorBoard

e_stop = EarlyStopping(monitor='loss',patience=5,mode='auto')
t_board = TensorBoard(log_dir='.\graph',histogram_freq=0,write_grads=True,write_graph=True, write_images=True)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

input1 = Input(shape=(x_train.shape[1],))


hid = Dense(5000,activation='relu')(input1)
hid = Dropout(0.1)(hid)
hid = Dense(200,activation='relu')(hid)
hid = Dropout(0.1)(hid)
hid = Dense(200,activation='relu')(hid)
output1 = Dense(10,activation='softmax')(hid)

model = Model(inputs=[input1], outputs=[output1])

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train,y_train,batch_size=500,epochs=100,validation_split=0.3, callbacks=[e_stop,t_board])

loss, acc = model.evaluate(x_test,y_test,batch_size=100)