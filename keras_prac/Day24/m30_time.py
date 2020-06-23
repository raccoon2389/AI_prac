from sklearn.feature_selection import SelectFromModel
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import r2_score
import time

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True, random_state = 66)
model = xgb.XGBRegressor()
model.fit(x_train,y_train)      
score = model.score(x_test,y_test)
# print(score)

thresholds = np.sort(model.feature_importances_)

# print(thresholds)
start = time.time()
for thres in thresholds: 
    selection = SelectFromModel(model, threshold=thres,prefit=True) #중요하지 않는 컬럼부터 하나씩 빼면서 트레이닝한다
                                        #median
    selection_x_train = selection.transform(x_train)
    model2 = xgb.XGBRegressor()
    model2.fit(selection_x_train,y_train)
    selection_x_test = selection.transform(x_test)
    
    score = model2.score(selection_x_test,y_test)
    
    # print(selection_x_train.shape)
    print(thres,score)

end = time.time()
print(end-start)
#그리드 서치 엮기
#데이콘 적용해라 71개 컬럼
#2번3번 파일 만들기
#메일제목 : 홍석규 10등

start2 = time.time()
for thres in thresholds: 
    selection = SelectFromModel(model, threshold=thres,prefit=True) #중요하지 않는 컬럼부터 하나씩 빼면서 트레이닝한다
                                        #median
    selection_x_train = selection.transform(x_train)
    model2 = xgb.XGBRegressor(n_jobs=3)
    model2.fit(selection_x_train,y_train)
    selection_x_test = selection.transform(x_test)
    
    score = model2.score(selection_x_test,y_test)
    
    # print(selection_x_train.shape)
    print(thres,score)

end2 = time.time()

print(end2-start2)
