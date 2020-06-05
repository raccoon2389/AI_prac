import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine = pd.read_csv('data/winequality-white.csv',sep=';',header=0)

y = wine['quality']
x = wine.drop('quality',axis=1)

print(x.shape)

# y레이블 축소
newlist = []
for i in list(y):
    if i<=4:
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else:
        newlist +=[2]
y = newlist

# print(y)

x_train, x_test, y_train, y_test = \
    train_test_split(x,y,test_size=0.2, shuffle=True)

model = RandomForestClassifier()
model.fit(x_train,y_train)
acc= model.score(x_test,y_test)
y_pred = model.predict(x_test)

print(f"acc_score : {accuracy_score(y_test,y_pred)}\nacc : {acc}")


