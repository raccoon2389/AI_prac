import os,sys
import numpy as np
from keras.models import Sequential, load_model, Input
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Day09.Keras39_split import split_X

#1 데이터
a = np.array(range(1,11))
size = 5                # time_steps= 5

#데이터 자르기

dataset = split_X(a,size)
x_train = dataset[:,:-1]
y_train = dataset[:,-1]
print(x_train)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)

print(x_train)

model = load_model("./keras_prac/model/save_keras_45.h5")

model.summary()

model.add(Dense(1, name= 'new'))
model.compile(optimizer='adam', loss = 'mse')

e_stop = EarlyStopping(monitor='loss',patience=40,mode='auto')

model.fit(x_train,y_train,batch_size=1,epochs=10000,callbacks=[e_stop])

mse = model.evaluate(x_train, y_train, batch_size=1)

print("MSE : ",mse)
x_predict = np.array([10,11,12,13])
x_predict = x_predict.reshape(1,4,1)
y_predict = model.predict(x_predict,batch_size=1)

print(x_predict,y_predict)

