# !과적합을 피하는법! No.2
# 1. 훈련 데이터를 늘린다
# 2. 피처수를 늘린다
# 3. 레귤러라리 제이션

#regulization
'''
L1 규제 : 가중치의 절대값 합
regularizer.l1(l=0.01)

L2 규제 : 가중치의 제곱의 합
regularizer.l2(l=0.01)

loss = L1*reduce_sum(abs(x))
loss = L2*reduce_sum(square(x))

'''

import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout,Input
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.optimizers import Adam



e_stop = EarlyStopping(monitor='loss',patience=5,mode='auto')
modelpath = "./keras_prac/model/{epoch:02d}--{val_loss:.4f}.hdf5"
m_check = ModelCheckpoint(filepath=modelpath, monitor = 'val_loss',save_best_only=True)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# scalar = StandardScaler()
# scalar.fit(x_train)
# x_train = scalar.transform(x_train)
# x_test = scalar.transform(x_test)

#데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
print(x_train.shape)

#데이터 전처리 2. 정규화
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

model = Sequential()
model.add(Conv2D(32,kernel_size=3, padding='same',activation='relu',input_shape=(32,32,3)))
# model.add(Dropout(0.3))

model.add(Conv2D(32,kernel_size=3, padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(32*2,kernel_size=3, padding='same',activation='relu'))
model.add(Dropout(0.3))

model.add(Conv2D(32*2,kernel_size=3, padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
# model.add(Dropout(0.3))

model.add(Conv2D(32*4,kernel_size=3, padding='same',activation='relu'))
model.add(Dropout(0.3))

model.add(Conv2D(32*4,kernel_size=3, padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
# model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(optimizer=Adam(1e-4),loss='sparse_categorical_crossentropy', metrics=['acc'])
hist = model.fit(x_train,y_train,batch_size=32,epochs=30,validation_split=0.4)

loss_acc = model.evaluate(x_test,y_test,batch_size=100)

print(loss_acc)



loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print("acc : ", acc)
# print("val_acc : ",val_acc)
# print("loss_acc : ",loss_acc)

plt.figure(figsize=(10,6))

plt.subplot(2,1,1) # 한창에 여러 표를 그린다
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') #
# plt.plot(hist.history['acc'])x
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='loss')
# plt.plot(hist.history['val_acc'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss', 'val_loss'])
plt.legend(loc='upper right')
plt.grid()

plt.subplot(2,1,2)
# plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_loss'])
plt.plot(hist.history['val_acc'])
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.grid()
plt.show()




print(loss,acc)




loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print("acc : ", acc)
# print("val_acc : ",val_acc)
# print("loss_acc : ",loss_acc)

plt.figure(figsize=(10,6))

plt.subplot(2,1,1) # 한창에 여러 표를 그린다
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') #
# plt.plot(hist.history['acc'])x
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='loss')
# plt.plot(hist.history['val_acc'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss', 'val_loss'])
plt.legend(loc='upper right')
plt.grid()

plt.subplot(2,1,2)
# plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_loss'])
plt.plot(hist.history['val_acc'])
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.grid()
plt.show()




print(loss,acc)
