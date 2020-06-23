from sklearn.feature_selection import SelectFromModel
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import r2_score
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True, random_state = 66)
model = xgb.XGBRegressor(n_estimators=1000,learning_rate=0.01)
model.fit(x_train,y_train,verbose=True,eval_metric='rmse',
eval_set=[(x_train,y_train),(x_test,y_test)],early_stopping_rounds=20)

result = model.evals_result()
print(result)
y_pred = model.predict(x_test)
score = r2_score(y_test,y_pred)
print(score)