import numpy as np
from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

'''
linear_svc란 퍼셉트론을 가장 간단하게 선형적으로 해결
직선을 그어서 분류를 한다
분류 경계에 위치한 샘플에 의해 전적으로 결정되므로 그 샘플을 서포트 백터라 한다.

'''

#1. 데이터
x_data = np.array([[0,0],[1,0],[0,1],[1,1]])
y_data = np.array([[0],[1],[1],[0]])

#2 모델
# model = SVC()
model = Sequential()# n네이버 
model.add(Dense(5,input_dim=2,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
#polynomial feature 이용해도 가능하다

model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy')


#3. 훈련
model.fit(x_data,y_data,batch_size=1,epochs=700)

#4. 평가예측
x_test = np.array([[0,0],[1,0],[0,1],[1,1]])
y_predict = model.predict(x_test)
y_predict = np.array(y_predict>0.5).astype('int')
print(y_predict)
acc= accuracy_score(y_data,y_predict)

print(x_test,'의 예측결과 : ',y_predict)
print("acc = ",acc)