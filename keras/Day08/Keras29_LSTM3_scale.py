'''
LSTM 이란

거대한 컨데어 벨트같은 cell state가 존재한다.
이 cell state가 
1. cell state는 반복모듈(유닛)을 단방향으로 이동하며 각 유닛에 들어가서 input과 h(t-1)[전 결과]를 concentate하고 Wf로 연산후 시그모이드를 취해 전 유닛으로부터 흘러 들어온 cell state와 곱하고 그것을 그대로 흘려주어서 전 cell state를 얼마나 반영할 것 인지를 정한다
2. 어느 방향으로 업데이트 할지 정하는 C 와 얼마나 가야할지 정하는 i와 연산후 cell state에 반영한다 (이 시점에서 새로운 cell state로 update된다. 즉 C(t-1)->C(t))
3. concentated한 데이터를 o와 연산하고 시그모이드 한다음 cell state과 곱한값여 cell state에서 어떤 데이터를 뽑아서 output에 내보낼지(h) 결정한다.

전 유닛으로 부터 흘러들어온 cell state를 보존하고 이용 할지말지 정하는 f, cell state의 update에 대한 방향성(C)와 세기(i), output에 어떤 data를 내보낼지 결정하는 o 이렇게 반복모듈 안에 4개의 layer 와 그안의 weight 또 각각 weight의 bias가 존재한다

그러므로 총 파라미터의 갯수는 (입력 차원 + 반복 모듈갯수 + bias(1) )*(반복 모듈갯수)*(layer안의 weight 갯수=4)이 된다.

출처 : http://colah.github.io/posts/2015-08-Understanding-LSTMs/
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.callbacks import EarlyStopping

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("shape of x : ", x.shape)


x = x.reshape(x.shape[0],x.shape[1],1)
print("shape of x : ", x.shape)


#2 모델 구성f
model = Sequential()
model.add(LSTM(200,activation='tanh',input_shape=(3,1)))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1))

model.summary() # LSTM 파라미터 폭증 이유 알아오기

# 실행
model.compile(optimizer='adam', loss='mse')

e_stop = EarlyStopping(monitor='loss',mode='auto',patience=15)
model.fit(x,y,epochs=10000,callbacks=[e_stop])

x_input = np.array([50,60,70])
x_input = x_input.reshape(1,3,1)

print(x_input)

yhat = model.predict(x_input)
print(yhat)
