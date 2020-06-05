import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold,cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')
#1. 데이터
iris = pd.read_csv('data/csv/iris.csv',header=0,sep=',')

x = iris.iloc[:,0:4]
y = iris.iloc[:,4]


x_train, x_test, y_train, y_test = \
    train_test_split(x,y,test_size=0.2, random_state=66, shuffle=True)


parameters = [
    {"C" : [1,10,100,1000],"kernel": ["linear","rbf","sigmoid"]},
    {"C" : [1,10,100,1000],"kernel": ["rbf"],"gamma" : [0.001,0.0001]},
    {"C" : [1,10,100,1000],"kernel": ["sigmoid" ],"gamma" : [0.001,0.0001]}
]

kf = KFold(n_splits=5, shuffle=True)

model = GridSearchCV(SVC(),param_grid=parameters,cv = kf)
model.fit(x_train,y_train)
print("최적의 매개변수 : ", model.best_params_)
score = model.score(x_test,y_test)

print(score)