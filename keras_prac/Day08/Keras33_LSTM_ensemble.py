import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM, Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from keras.layers.merge import concatenate
#1. 데이터
x1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
x2 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[9,10,11],[10,11,12],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x1 = x1.reshape(x1.shape[0],3,1)
x2 = x2.reshape(x2.shape[0],3,1)
print("shape of x : ", x1.shape)
print("shape of x : ", x2.shape)


#2 모델 구성f

input1 = Input(shape=(3,1))
lstm1_1 = LSTM(450, activation='tanh')(input1)

input2 = Input(shape=(3,1))
lstm2_1 = LSTM(450,activation='tanh')(input2)

merge = concatenate([lstm1_1,lstm2_1])

middle1 = Dense(10,activation='relu')(merge)
output2 = Dense(20,activation='relu')(middle1)
output3 = Dense(1)(output2)


model = Model(inputs=[input1,input2],outputs=[output3])



model.summary() 

# 실행
model.compile(optimizer='adam', loss='mse')

e_stop = EarlyStopping(monitor='loss',mode='auto',patience=50)
model.fit([x1,x2],y,epochs=10000,callbacks=[e_stop])

x1_predict = np.array([55,65,75])
x1_predict = x1_predict.reshape(1,3,1)
x2_predict = np.array([65,75,85])
x2_predict = x2_predict.reshape(1,3,1)

print(x2_predict)

yhat = model.predict([x1_predict,x2_predict])
print(yhat)