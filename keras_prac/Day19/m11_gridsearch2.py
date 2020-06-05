import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold,cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')
#1. 데이터

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = \
    train_test_split(x,y,test_size=0.2, random_state=66, shuffle=True)


# x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
# x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])

print((x_train.shape,x_test.shape) ,(y_train.shape,y_test.shape))


parameters = [
    {"n_estimators" : [10,100],"min_samples_leaf": [3,5,7,9],"min_samples_split":[2,3,5,7,9] ,"max_features" : ["auto", "sqrt", "log2"], "criterion" : ["gini", "entropy"], "max_depth": [None,10,100]}#
]

kf = KFold(n_splits=5, shuffle=True)

model = GridSearchCV(RandomForestClassifier(),param_grid=parameters,cv = kf,n_jobs = -1)
model.fit(x_train,y_train)
print("최적의 매개변수 : ", model.best_estimator_)
score = model.score(x_test,y_test)

print(score)