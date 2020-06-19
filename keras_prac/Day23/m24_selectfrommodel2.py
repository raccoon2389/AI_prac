from sklearn.feature_selection import SelectFromModel
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import r2_score
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True, random_state = 66)
model = xgb.XGBRegressor()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
# print(score)

thresholds = np.sort(model.feature_importances_)

# print(thresholds)

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

#그리드 서치 엮기
#데이콘 적용해라 71개 컬럼
#메일제목 : 홍석규 10등