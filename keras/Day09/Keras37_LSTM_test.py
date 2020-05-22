
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
# 실습 : LSTM 레이어를 5개 이상 엮어서 Dense 결과를 이겨내시오
#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("shape of x : ", x.shape)


x = x.reshape(x.shape[0],3,1)
print("shape of x : ", x.shape)


#2 모델 구성f
input1 = Input(shape=(3,1))
lstm1_1 = LSTM(7,activation='tanh',return_sequences=True)(input1) # 배열을 2d가 아닌 3d로 출력하게 만든다.
lstm1_2 = LSTM(7,activation='tanh',return_sequences=True)(lstm1_1)
lstm1_2 = LSTM(7,activation='tanh',return_sequences=True)(lstm1_2)
lstm1_2 = LSTM(7,activation='tanh',return_sequences=True)(lstm1_2)
lstm1_2 = LSTM(7,activation='tanh')(lstm1_2)
output3 = Dense(10,activation='relu')(lstm1_2)
output3 = Dense(5,activation='relu')(lstm1_2)
output3 = Dense(3,activation='relu')(lstm1_2)

output3 = Dense(1,activation='relu')(output3)



model = Model(inputs=[input1],outputs=[output3])


model.summary() 

# 실행
model.compile(optimizer='adam', loss='mse')

e_stop = EarlyStopping(monitor='loss',mode='auto',patience=50)
model.fit(x,y,epochs=10000,callbacks=[e_stop])

x_input = np.array([55,65,75])
x_input = x_input.reshape(1,3,1)
y_test = np.array([[85]])
print(x_input)

yhat = model.predict(x_input)
print(yhat)