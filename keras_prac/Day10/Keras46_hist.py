import os,sys
import numpy as np
from keras.models import Sequential, load_model, Input
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Day09.Keras39_split import split_X
import matplotlib.pyplot as plt

#1 데이터
a = np.array(range(1,101))
size = 5                # time_steps= 5

#데이터 자르기

dataset = split_X(a,size)
x_train = dataset[:,:-1]
y_train = dataset[:,-1]
print(x_train)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)

print(x_train)

model = load_model("./keras_prac/model/save_keras_45.h5")

model.add(Dense(50,activation='relu', name= 'new1'))
model.add(Dense(10,activation='relu', name= 'new2'))
model.add(Dense(10,activation='relu', name= 'new3'))
model.add(Dense(1, name= 'new4'))

model.summary()

model.compile(optimizer='adam', loss = 'mse',  metrics=['acc'])

e_stop = EarlyStopping(monitor='loss',patience=20,mode='auto')
hist = model.fit(x_train,y_train, validation_split=0.2,
                batch_size=1,epochs=100,
                callbacks=[e_stop])

plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['val_acc'])
plt.title('loss&acc')
plt.ylabel('loss,acc')
plt.xlabel('epoch')
plt.legend(['loss','acc', 'val_loss', 'val_acc'])
plt.show()
'''
mse = model.evaluate(x_train, y_train, batch_size=1)

print("MSE : ",mse)
x_predict = np.array([10,11,12,13])
x_predict = x_predict.reshape(1,4,1)
y_predict = model.predict(x_predict,batch_size=1)

print(x_predict,y_predict)
'''