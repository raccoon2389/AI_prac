'''
m28_eval2,3 만들기
1. 회귀
2. 이진분류
3. 다중분류

sfm에는 그래프 노우
1. eval에 'loss'와 다른 지표 1개 더 추가
2. earlystopping 적용
3. plot으로 그릴것.

4. 결과는 주석으로 소스 하단에 표시.


'''

from sklearn.feature_selection import SelectFromModel
import numpy as np
# import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True, random_state = 66)
model = LGBMClassifier()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
# print(score)

thresholds = np.sort(model.feature_importances_)

# print(thresholds)
models = [] # 빈 모델 배열 생성
res = np.array([]) #빈 결과값 배열 생성
for thres in thresholds: 
    selection = SelectFromModel(model, threshold=thres,prefit=True) #중요하지 않는 컬럼부터 하나씩 빼면서 트레이닝한다
                                        #median
    selection_x_train = selection.transform(x_train)
    model2 = LGBMClassifier(n_estimators=1000)
    selection_x_test = selection.transform(x_test)
    model2.fit(selection_x_train,y_train,verbose=True,eval_metric='mae',eval_set=[(selection_x_train,y_train),(selection_x_test,y_test)],early_stopping_rounds=20)
    result = model2.evals_result_
    score = model2.score(selection_x_test,y_test)
    shape = selection_x_train.shape
    models.append(model2) # 모델을 전부 배열에 저장
    print(thres,score)
    res = np.append(res,score)# 결과값을 전부 배열에 저장
print(res.shape)
best_idx = res.argmax() # 결과값에 최대값의 index 저장
score = res[best_idx]   # 위 인덱스 기반으로 점수호출
total_col = x_train.shape[1]-best_idx # 전체컬럼 계산
pickle.dump(models[best_idx],open(f"./model/breast_cancer--{score}--{total_col}--.model","wb") ) # 인덱스 기반으로 모델 저장
#[0.9211316262835174, 0.9222460247849769, 0.9236430220063924, 0.9262235025233723, 0.9372555799320916, 0.935001131522476, 0.9322758377604793, 0.9363292168940863, 0.9253299621181523, 0.9238701992987739, 0.9332629889746407, 0.7791269721195485, 0.7022318255202269]