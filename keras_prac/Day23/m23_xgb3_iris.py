# 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피쳐수를 줄인다.
# 3. regularization

from xgboost import XGBClassifier,plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from numpy.random import randint
dataset = load_iris()
x = dataset.data
y = dataset.target
# print(x.shape)        #(506, 13)

# print(y.shape)        #(506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True, random_state = 23)
param = [
    {"n_estimators" :[100,1000, 5000],
    'learning_rate': [0.1,0.01,0.001],
    'colsample_bytree': [1,0.8,0.5],
    'colsample_bylevel' : [1,0.8,0.5],

    'max_depth'  : [3,5,7,9],
    'n_jobs' : [-1]}
]


model = GridSearchCV(XGBClassifier(),param_grid=param,)

model.fit(x_train,y_train)
print('=====================================')
print(model.best_params_)
print('=====================================')

print(model.best_estimator_)
print('=====================================')
score = model.score(x_test,y_test)
print(score)
# plot_importance(model)
# plt.show()#
#0.966