#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18,19,20])

#2. model 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(1000, input_dim =1))
model.add(Dense(40))
model.add(Dense(1000))
model.add(Dense(40))


model.add(Dense(10))



# model.add(Dense(1000000))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])    # mse 평균제곱에러 (실제 데이터값 - 예측값)의 제곱 을 평균으로 나눈다. 
                                                                # 0.001의 오차만 나오더라도 mse 값이 매우 낮아짐 
                                                                # acc는 분류지표 mse는 회귀지표
                                                                # 분류 = '개', '고양이' 같은 딱딱 떨어지는 비연속적인 값 미리 설정한 y값이 고정됨 -> 자신이 설정한 y값 이외에 안나옴
                                                                # 회귀 = 1.54 , 10.01 같이 연속적인 값 
                                                                # metrics는 loss처럼 훈련에 영향은 주지 않고 계산한 값만 뱉어냄

model.fit(x_train,y_train, epochs=300, batch_size=1)

#4. evaluate,predict
loss, mse = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss,"\nmse : ",mse)

y_pred = model.predict(x_pred)
print(y_pred)