
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("shape of x : ", x.shape)


x = x.reshape(x.shape[0],3,1)
print("shape of x : ", x.shape)


#2 모델 구성f
input1 = Input(shape=(3,1))
lstm1_1 = LSTM(10,activation='tanh')(input1)

middle1 = Dense(20,activation='relu')(lstm1_1)
middle2 = Dense(20,activation='relu')(middle1)
middle3 = Dense(20,activation='relu')(middle2)

output1 = Dense(10,activation='relu')(middle3)
output2 = Dense(10,activation='relu')(output1)
output3 = Dense(1,activation='relu')(output2)


model = Model(inputs=[input1],outputs=[output3])


model.summary() 

# 실행
model.compile(optimizer='adam', loss='mse')

e_stop = EarlyStopping(monitor='loss',mode='auto',patience=20)
model.fit(x,y,epochs=10000,callbacks=[e_stop])

x_input = np.array([50,60,70])
x_input = x_input.reshape(1,3,1)
y_test = np.array([[80]])
print(x_input)

yhat = model.predict(x_input)
print(yhat)