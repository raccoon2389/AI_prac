#1. sequencial -> model = sequencial
#2. 함수형 -> 다수의 모델을 연계


# 1. 데이터

import numpy as np
# 열우선 행무시 x=np.array([range(1,101),range(311,411),range(100)]) 이렇게 쓰면 (3,100)이되어서 가로로 바꿔야함
x1 = np.array([range(1, 101), range(311, 411), range(100)])
y1 = np.array([range(711, 811),range(711, 811),range(711, 811)])

x2 = np.array([range(1, 101), range(311, 411), range(100)])
y2 = np.array([range(711, 811),range(711, 811),range(711, 811)])

from sklearn.model_selection import train_test_split

# train_split #
# split을하여 데이터를 나누어준다
# randomstate가 같게되면 같은 난수표를 사용하여서 shuffle을 안해주면 동일한 결과가 나온다
#


# (3,100) 배열을 (100,3)으로 변환

# transpose 이용
x1=np.transpose(x1)
y1=np.transpose(y1)
x2=np.transpose(x2)
y2=np.transpose(y2)

# scipy의 rotate함수를 이용하여 변환하는 방법
# from scipy.ndimage.interpolation import rotate


# x=rotate(x, angle=-90)
# y = rotate(y, angle = -90)
# print(x.shape, y.shape)

# ###### for loop를 이용한 자작 변환 함수 (2d matrics -90도 회전만 가능)
# shape = x.shape
# print(shape)
# x2 = np.zeros((shape[1],shape[0]))

# for r in range(100):
#     for c in range(3):
#         x2[r,c] = x[c,r]
# #####

x1_train, x1_test, y1_train, y1_test=train_test_split(
    x1, y1,
    random_state=66,
    test_size=0.2,
    shuffle=True)

x2_train, x2_test, y2_train, y2_test=train_test_split(
    x2, y2,
    random_state=66,
    test_size=0.2,
    shuffle=True)

# x_train, x_val, y_train, y_val = train_test_split(
#     x_train,y_train,
#     random_state=99,
#     test_size=0.25,
#     shuffle=Ture)

# 다른방법으로 이것이 있다.
# xa = np.arange(10.0)
# np.split(xa, [ int(len(xa)*0.6), int(len(xa)*0.8)])

# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]

# y_train = x[:60]
# y_val = x[60:80]
# y_test = x[80:]

# 2. model 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#선형모델
# model=Sequential()

# model.add(Dense(5, input_dim=3))
# model.add(Dense(4))
# model.add(Dense(1))

#함수 모델

input1 = Input(shape=(3,))

dense1_1 = Dense(5,activation='relu', name='bitking1')(input1)
dense1_2 = Dense(4,activation='relu', name='bitking2')(input1)
dense1_3 = Dense(4,activation='relu', name='bitking3')(input1)

input2 = Input(shape=(3,))

dense2_1 = Dense(5,activation='relu', name='bitking4')(input1)
dense2_2 = Dense(4,activation='relu', name='bitking5')(input1)
dense2_3 = Dense(4,activation='relu', name='bitking6')(input1)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3,dense2_3] )

middle1 = Dense(30)(merge1)
middle2 = Dense(5)(middle1)
middle3 = Dense(7)(middle2)

############# output model ##############
output1 =  Dense(30)(middle3)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(30)(middle3)
output2_2 = Dense(7)(output2)
output2_3 = Dense(3)(output2_2)

model = Model(inputs=[input1,input2] ,outputs=[output1_3,output2_3])
model.summary()


# 3. 훈련
# mse 평균제곱에러 (실제 데이터값 - 예측값)의 제곱 을 평균으로 나눈다.
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
                                                                # 0.001의 오차만 나오더라도 mse 값이 매우 낮아짐
                                                                # acc는 분류지표 mse는 회귀지표
                                                                # 분류 = '개', '고양이' 같은 딱딱 떨어지는 비연속적인 값 미리 설정한 y값이 고정됨 -> 자신이 설정한 y값 이외에 안나옴
                                                                # 회귀 = 1.54 , 10.01 같이 연속적인 값
                                                                # metrics는 loss처럼 훈련에 영향은 주지 않고 계산한 값만 뱉어냄

# epoch = 훈련 횟수 ; 일정수 이상의 훈련을 반복하면 과적합(over-fitting)이 일어나게 된다.
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=10, batch_size=1, verbose=1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       ) # verbose = 0: 아무것도 안보임 1: 기본값 2: 프로그레스바 삭제 3: epoch만
                                                                                    # validation set = train set 중 일부를 떼와서 train으로 훈련후 검증한다
                                                                                    # fit하는 과정에 반영이 된다. W 값 최적화에 도움이 됨
                                                                                    # test는 최종 확인만 하므로 fit 과정에 영향을 주지 않음
                                                                                    # validation accuracy 낮으면 overfit 가능성이 높다
                                                                                    # validation test set과 다르다
                                                                                    # validation_split 인수를 이용해 자체적으로 가능하다

# 4. evaluate,predict
loss, mse,x,y,z = model.evaluate([x1_test,x2_test], [y1_test,y2_test], batch_size=1)
print("loss : ", loss, "\nmse : ", mse,x,y,z)

y1_predict , y2_predict=model.predict([x1_test,x2_test])
print(x1_test,y1_predict)
###통합하는 방법##
# y_predict = np.append(y_predict[0],y_predict[1])
# y_test = np.append(y1_test,y2_test)

from sklearn.metrics import mean_squared_error
# RMSE 구하기
def RMSE(y_test, y_predict):  # rmse = mse를 root 취한값 크기에 의존적인게 단점이다. mse값이 1이하로 작아졌을때 크기가 커지고 1이상일때 크기가 작아진다. 1이하일때 보기 더 편해짐
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE1 = RMSE(y1_test, y1_predict) 
RMSE2 = RMSE(y2_test,y2_predict)
print("RMSE : ",(RMSE1+RMSE2)/2 )


# R2만들기

    # R2 데이터와 평균이 얼마나 떨어져있는지에 대한 지표이다.
    # RMSE는  y-y' 잔차(residual= 실제 데이터와 예측값 또는 오차) 사이를 설명한다.
    # 모델 그래프와 평균값 사이의 차를 편차(regression)라고 한다.
    # R2 = 1 - 편차^2/편차^2 이므로 오차가 편차보다 더 커지게 되면 음수가 나온다
from sklearn.metrics import r2_score
r2_1=r2_score(y1_test, y1_predict)
r2_2=r2_score(y2_test, y2_predict)
print("R2 : ", (r2_1+r2_2)/2)
