import os,sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Day09.Keras39_split import split_X



model = Sequential()
model.add(LSTM(2, activation='tanh',input_shape = (4,1)))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))
model.save("./keras_prac/model/save_keras_45.h5")
'''

model.compile(optimizer='adam', loss = 'mse')

e_stop = EarlyStopping(monitor='loss',patience=40,mode='auto')

model.fit(x_train,y_train,batch_size=1,epochs=10000,callbacks=[e_stop])

mse = model.evaluate(x_train, y_train, batch_size=1)

print("MSE : ",mse)
x_predict = np.array([10,11,12,13])
x_predict = x_predict.reshape(1,4,1)
y_predict = model.predict(x_predict,batch_size=1)

print(x_predict,y_predict)

'''