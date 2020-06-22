'''
m28_eval2,3 만들기
1. 회귀
2. 이진분류
3. 다중분류

1. eval에 'loss'와 다른 지표 1개 더 추가
2. earlystopping 적용
3. plot으로 그릴것.

4. 결과는 주석으로 소스 하단에 표시.


'''

from sklearn.feature_selection import SelectFromModel
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True, random_state = 66)
model = xgb.XGBClassifier()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
# print(score)

thresholds = np.sort(model.feature_importances_)

# print(thresholds)
res = []
for thres in thresholds: 
    selection = SelectFromModel(model, threshold=thres,prefit=True) #중요하지 않는 컬럼부터 하나씩 빼면서 트레이닝한다
                                        #median
    selection_x_train = selection.transform(x_train)
    model2 = xgb.XGBClassifier(n_estimators=1000)
    selection_x_test = selection.transform(x_test)
    model2.fit(selection_x_train,y_train,verbose=True,eval_metric='merror',eval_set=[(selection_x_train,y_train),(selection_x_test,y_test)],early_stopping_rounds=20)
    result = model2.evals_result()
    

    score = model2.score(selection_x_test,y_test)
    
    # print(selection_x_train.shape)
    print(thres,score)
    res.append(score)
# plt.show()
print(res)
#validation_0-merror:-0.32500     validation_1-merror:-0.36667