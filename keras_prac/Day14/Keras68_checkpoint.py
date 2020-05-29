import numpy as np
import sys,os
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D,Flatten, Dropout #Dropout 중간에 노드들을 일정비율을 지운다.
from keras.callbacks import EarlyStopping,ModelCheckpoint
e_stop = EarlyStopping(monitor='loss',patience=5,mode='auto')


m_check = ModelCheckpoint(filepath=".\keras_prac\model\{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

#데이터 전처리 2. 정규화
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1).astype('float32')/255.

model = Sequential()

model.add(Conv2D(64,(2,2),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
# model.add(Conv2D(10,(2,2),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.3))
# model.add(Conv2D(10,(3,3),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
hist = model.fit(x_train,y_train,batch_size=100,epochs=10,validation_split=0.2, callbacks=[e_stop,m_check])

loss_acc = model.evaluate(x_test,y_test,batch_size=100)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print("acc : ", acc)
print("val_acc : ",val_acc)
print("loss_acc : ",loss_acc)

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