import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('data/csv/boston_house_prices.csv',header=1,sep=',')

x = iris.iloc[:,0:13].values
y = iris.iloc[:,13].values
y = y.reshape(-1,1)

# print(y)

x_train, x_test, y_train, y_test = \
    train_test_split(x,y,test_size=0.2, random_state=66, shuffle=True)

model = all_estimators(type_filter='regressor')
for (name, algorithm) in model:
    model = algorithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name,'의 정답률 : ', model.score(x_test,y_test))

import sklearn
print(sklearn.__version__)
