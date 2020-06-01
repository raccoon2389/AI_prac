import numpy as np
from sklearn.datasets import load_boston
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout,Input
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #robust와 pca로 이상치 제거 가능 PCA(column 압축)하기전에 반드시 standard scaler

e_stop = EarlyStopping(monitor='loss',patience=5,mode='auto')
modelpath = "./keras_prac/model/{epoch:02d}--{val_loss:.4f}.hdf5"
m_check = ModelCheckpoint(filepath=modelpath, monitor = 'val_loss',save_best_only=True)
# dataset = load_boston()
# x = dataset.data
# y = dataset.target
x, y = load_boston(return_X_y=True)
print(y.shape)
x_train, x_test , y_train, y_test = train_test_split(x,y,shuffle=True,test_size = 0.2)


scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)

pc = PCA(n_components=8)
pc.fit(x_train)
x_train = pc.transform(x_train)
x_test = pc.transform(x_test)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

x_train = x_train.reshape(x_train.shape[0],4,2,1)
x_test = x_test.reshape(x_test.shape[0],4,2,1)
input1 = Input(shape=(x_train.shape[1],x_train.shape[2],1))



hid = Conv2D(10,(2,2),padding='same',activation='relu')(input1)
# hid = MaxPooling2D(pool_size=(2,2))(hid)
hid = Dropout(0.4)(hid)
hid = Conv2D(10,(2,2),padding='same',activation='relu')(hid)
# hid = MaxPooling2D(pool_size=(2,2))(hid)
hid = Dropout(0.3)(hid)

hid = Flatten()(hid)
output1 = Dense(200,activation='relu')(hid)
output1 = Dense(1)(hid)


model = Model(inputs=[input1], outputs=[output1])

model.summary()

model.compile(optimizer='adam',loss='mse', metrics=['acc'])
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
