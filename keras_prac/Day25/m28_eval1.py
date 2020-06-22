from sklearn.feature_selection import SelectFromModel
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
dataset = load_boston()
x = dataset.data
y = dataset.target

param = {
    'n_estimators':np.random.randint(100,1000,30).tolist(),
    'max_depth':np.random.randint(2,20,10).tolist(),
    'learning_rate':np.linspace(0.01,1.5,10).tolist()

}

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True, random_state = 66)
xg = xgb.XGBRegressor(gpu_id=0,tree_method='gpu_hist')
model = RandomizedSearchCV(xg,param,n_iter=5,)
model.fit(x_train,y_train,verbose=True,eval_metric=['logloss','rmse'],
eval_set=[(x_train,y_train),(x_test,y_test)],early_stopping_rounds=20)

result = model.result
print(result)
y_pred = model.predict(x_test)
score = r2_score(y_test,y_pred)
print(score)

epochs = len(result)
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result,label='Train')
ax.plot(x_axis, result,label='Eval')
ax.legend()
plt.ylabel('logloss')
plt.title('XGboost log loss')
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(x_axis, result['validation_0']['rmse'],label='Train')
# ax.plot(x_axis, result['validation_1']['rmse'],label='Eval')
# ax.legend()
# plt.ylabel('rmse')
# plt.title('XGboost rmse')
plt.show()