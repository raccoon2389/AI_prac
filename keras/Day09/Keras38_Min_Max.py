import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler     # StandardScaler(X)(표준화): 평균이 0과 표준편차가 1이 되도록 변환. (x-x.ave)/(표준편차)
                                                                                            # MinMaxScaler(X)(정규화): 최대값이 각각 1, 최소값이 0이 되도록 변환 (x-x.min)/(x.max-x.min)
                                                                                            # RobustScaler(X): 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환.
                                                                                            # MaxAbsScaler(X): 0을 기준으로 절대값이 가장 큰 수가 1또는 -1이 되도록 변환

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[2000,3000,4000],[3000,4000,5000],[4000,5000,6000],[100,200,300]])
y = np.array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400])
x_predict = np.array([55,65,75])
#scalar = StandardScaler()
scalar = MinMaxScaler()
scalar.fit(x)
x = scalar.transform(x)
print("shape of x : ", x)

x_predict = x_predict.reshape(1,3)
x_predict = scalar.transform(x_predict)

x = x.reshape(x.shape[0],3,1)

print("shape of x : ", x.shape)

input1 = Input(shape=(3,1))
lstm1_1 = LSTM(500,activation='tanh')(input1) # 배열을 2d가 아닌 3d로 출력하게 만든다.
lstm1_1 = Dense(10,activation='relu')(lstm1_1)
lstm1_1 = Dense(10,activation='relu')(lstm1_1)
lstm1_1 = Dense(10,activation='relu')(lstm1_1)
lstm1_1 = Dense(10,activation='relu')(lstm1_1)
lstm1_1 = Dense(10,activation='relu')(lstm1_1)
lstm1_1 = Dense(10,activation='relu')(lstm1_1)
lstm1_1 = Dense(10,activation='relu')(lstm1_1)
lstm1_1 = Dense(10,activation='relu')(lstm1_1)
lstm1_1 = Dense(10,activation='relu')(lstm1_1)
lstm1_1 = Dense(10,activation='relu')(lstm1_1)
lstm1_1 = Dense(10,activation='relu')(lstm1_1)
lstm1_1 = Dense(10,activation='relu')(lstm1_1)
output3 = Dense(1,activation='relu')(lstm1_1)



model = Model(inputs=[input1],outputs=[output3])


model.summary() 

# 실행
model.compile(optimizer='adam', loss='mse')

e_stop = EarlyStopping(monitor='loss',mode='auto',patience=100)
model.fit(x,y,epochs=10000,callbacks=[e_stop], batch_size=1)


x_predict = x_predict.reshape(1,3,1)

y_test = np.array([[80]])
print(x_predict)

yhat = model.predict(x_predict)
print(yhat)
