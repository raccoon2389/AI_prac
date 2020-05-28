import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, MaxPool2D,Flatten, Dropout, Input #Dropout 중간에 노드들을 일정비율을 지운다.
from keras.callbacks import EarlyStopping, TensorBoard
e_stop = EarlyStopping(monitor='loss',patience=5,mode='auto')
t_board = TensorBoard(log_dir='.\graph',histogram_freq=0,write_grads=True,write_graph=True, write_images=True)
(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train[0])
print(y_train[0])

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

# plt.imshow(x_train[0],'gray')
# plt.show()

#데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

#데이터 전처리 2. 정규화
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1).astype('float32')/255.

input1 = Input(shape=(x_train.shape[1],x_train.shape[2],1))

hid = Conv2D(50,(2,2),activation='relu')(input1)
hid = MaxPool2D(pool_size=(2,2))(hid)
hid = Dropout(0.1)(hid)
hid = Conv2D(10,(2,2),activation='relu')(hid)
hid = MaxPool2D(pool_size=(2,2))(hid)
hid = Dropout(0.1)(hid)

hid = Flatten()(hid)
output1 = Dense(10,activation='softmax')(hid)

model = Model(inputs=[input1], outputs=[output1])

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train,y_train,batch_size=100,epochs=100,validation_split=0.2, callbacks=[e_stop,t_board])

loss, acc = model.evaluate(x_test,y_test,batch_size=100)

print(loss,acc)

#mse = 0.04351499668984616 acc = 0.98580002784729