
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]]) #(13,3)
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("shape of x : ", x.shape)


# x = x.reshape(x.shape[0],3,1)
print("shape of x : ", x.shape)


#2 Dense 모델 구성

model = Sequential()

model.add(Dense(10,activation='relu',input_shape=(3,)))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))
model.summary() 

# 실행
model.compile(optimizer='adam', loss='mse')

e_stop = EarlyStopping(monitor='loss',mode='auto',patience=20)
model.fit(x,y,epochs=10000,callbacks=[e_stop])

x_predict = np.array([55,65,75])
x_predict = x_predict.reshape(1,3)
y_test = np.array([[80]])
print(x_predict)

yhat = model.predict(x_predict)
print(yhat)