from sklearn.feature_selection import SelectFromModel
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target      

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True, random_state = 66)
model = xgb.XGBClassifier(n_estimators=400,learning_rate=0.01)
model.fit(x_train,y_train,verbose=True,eval_metric=['error'],
eval_set=[(x_train,y_train),(x_test,y_test)],early_stopping_rounds=20)
import joblib
result = model.evals_result()
print(result)
y_pred = model.predict(x_test)
score = accuracy_score(y_test,y_pred)
print(score)
model.save_model("./model/cancer.xgb.model")
model2 = xgb.XGBClassifier()
model2.load_model("./model/cancer.xgb.model")
# dump(model,("./model/cancer.joblib.dat"))
# model2 = joblib.load(("./model/cancer.joblib.dat"))
# pickle.dump(model,open("./model/cancer.pickle.dat","wb"))

# model2 = pickle.load(open("./model/cancer.pickle.dat","rb"))
result = model2.evals_result()
print(result)
y_pred = model2.predict(x_test)
score = accuracy_score(y_test,y_pred)
print(score)
