import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data/winequality-white.csv',sep=';',header=0)

print(dataset.head()) 
# null = dataset.isnull().sum()
# print(null) # 0

print(dataset)

x = dataset.loc[:,"fixed acidity":"alcohol"].values



y = dataset.loc[:,"quality"].values.astype('int')

x_train, x_test, y_train, y_test = \
    train_test_split(x,y,test_size=0.2, shuffle=True)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#############       KNeighbors      ###############
'''
neigh= KNeighborsClassifier()
neigh.fit(x_train,y_train)
score = neigh.score(x_test,y_test)
print("Regressor - R2 score : ",score)
'''
##################################################


############        RandomForest    ##############

clasi = RandomForestClassifier()
clasi.fit( x_train,y_train)
y_pred = clasi.predict(x_test)
acc = clasi.score(x_test,y_test)
print("Classifier - accuracy : ",acc)
##################################################

###########         SVC             ###############




