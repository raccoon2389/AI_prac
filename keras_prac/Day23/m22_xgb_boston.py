# 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피쳐수를 줄인다.
# 3. regularization

from xgboost import XGBRegressor,plot_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
# print(x.shape)        #(506, 13)

# print(y.shape)        #(506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True, random_state = 23)

n_estimator = 50000
learning_rate = 0.075
colsample_bytree =1
colsample_bylevel =1

max_depth  = 2
n_jobs = -1

model = XGBRegressor(max_depth = max_depth,n_estimators=n_estimator, learning_rate=learning_rate,n_jobs=n_jobs,colsample_bylevel=colsample_bylevel,colsample_bytree=colsample_bytree)
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print(score)
plot_importance(model)
# plt.show()