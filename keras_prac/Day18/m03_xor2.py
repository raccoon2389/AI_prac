from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
'''
linear_svc란 퍼셉트론을 가장 간단하게 선형적으로 해결
직선을 그어서 분류를 한다
분류 경계에 위치한 샘플에 의해 전적으로 결정되므로 그 샘플을 서포트 백터라 한다.

'''
#1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

#2 모델
model = SVC()
#polynomial feature 이용해도 가능하다

#3. 훈련
model.fit(x_data,y_data)

#4. 평가예측
x_test = [[0,0],[1,0],[0,1],[1,1]]
y_predict = model.predict(x_test)
acc= accuracy_score([0,1,1,0],y_predict)

print(x_test,'의 예측결과 : ',y_predict)
print("acc = ",acc)