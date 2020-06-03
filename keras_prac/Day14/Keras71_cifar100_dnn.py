import numpy as np
from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Dense,Input,Dropout
from keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
e_stop = EarlyStopping(monitor='loss',patience=5,mode='auto')
modelpath = "./keras_prac/model/{epoch:02d}--{val_loss:.4f}.hdf5"
m_check = ModelCheckpoint(filepath=modelpath, monitor = 'val_loss',save_best_only=True)
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape)
# plt.imshow(x_train[1])
# plt.show()

#데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(x_train.shape)

#데이터 전처리 2. 정규화
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3]).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3]).astype('float32')/255.


input1 = Input(shape=(x_train.shape[1],))


hid = Dense(40000,activation='relu')(input1)
hid = Dropout(0.3)(hid)
hid = Dense(20000,activation='relu')(input1)
hid = Dropout(0.3)(hid)
hid = Dense(300,activation='relu')(input1)
hid = Dropout(0.3)(hid)
output1 = Dense(100,activation='softmax')(hid)

model = Model(inputs=[input1], outputs=[output1])

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
hist = model.fit(x_train,y_train,batch_size=500,epochs=100,validation_split=0.2, callbacks=[e_stop,m_check])

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