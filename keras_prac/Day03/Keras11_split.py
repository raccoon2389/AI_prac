#1. 데이터
import numpy as np
x=np.arange(1,101)
y=np.arange(101,201)

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

y_train = x[:60]
y_val = x[60:80]
y_test = x[80:]

#2. model 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(50, input_dim =1))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])    # mse 평균제곱에러 (실제 데이터값 - 예측값)의 제곱 을 평균으로 나눈다. 
                                                                # 0.001의 오차만 나오더라도 mse 값이 매우 낮아짐 
                                                                # acc는 분류지표 mse는 회귀지표
                                                                # 분류 = '개', '고양이' 같은 딱딱 떨어지는 비연속적인 값 미리 설정한 y값이 고정됨 -> 자신이 설정한 y값 이외에 안나옴
                                                                # 회귀 = 1.54 , 10.01 같이 연속적인 값 
                                                                # metrics는 loss처럼 훈련에 영향은 주지 않고 계산한 값만 뱉어냄

model.fit(x_train,y_train, epochs=110, batch_size=1, validation_data=(x_val,y_val)) # epoch = 훈련 횟수 ; 일정수 이상의 훈련을 반복하면 과적합(over-fitting)이 일어나게 된다. 
                                                                                    # validation set = train set 중 일부를 떼와서 train으로 훈련후 검증한다
                                                                                    # fit하는 과정에 반영이 된다. W 값 최적화에 도움이 됨
                                                                                    # test는 최종 확인만 하므로 fit 과정에 영향을 주지 않음
                                                                                    # validation accuracy 낮으면 overfit 가능성이 높다
                                                                                    # validation test set과 다르다

#4. evaluate,predict
loss, mse = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss,"\nmse : ",mse)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error
#RMSE 구하기
def RMSE(y_test,y_predict):           #rmse = mse를 root 취한값 크기에 의존적인게 단점이다. mse값이 1이하로 작아졌을때 크기가 커지고 1이상일때 크기가 작아진다. 1이하일때 보기 더 편해짐
    return np.sqrt(mean_squared_error(y_test,y_predict))

print("RMSE : ",RMSE(y_test,y_predict))


# R2만들기

    # R2 데이터와 평균이 얼마나 떨어져있는지에 대한 지표이다. 
    # RMSE는  y-y' 잔차(residual= 실제 데이터와 예측값 또는 오차) 사이를 설명한다.
    # 모델 그래프와 평균값 사이의 차를 편차(regression)라고 한다. 
    # R2 = 1 - 편차^2/편차^2 이므로 오차가 편차보다 더 커지게 되면 음수가 나온다
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

