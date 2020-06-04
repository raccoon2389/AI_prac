import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import r2_score,accuracy_score

data = load_breast_cancer()
x = data.data
y = data.target

# y = np_utils.to_categorical(y)

lsvc = LinearSVC()
lsvc.fit(x,y)
l_pred = lsvc.predict(x)
score = lsvc.score(x,y)
print(score)

svc = SVC()
svc.fit(x,y)
l_pred = svc.predict(x)
score = svc.score(x,y)
print(score)


kne = KNeighborsClassifier(n_neighbors=1)
kne.fit(x,y)
k_pred = kne.predict(x)
# k_pred = np_utils.to_categorical(k_pred)
acc1 = accuracy_score(y,k_pred)
# print(k_pred)
print(acc1)



ran_fo = RandomForestClassifier()
ran_fo.fit(x,y)
ran_pred = ran_fo.predict_proba(x)
new_pred = np.array([])
for i in range(len(ran_pred)):
    max = ran_pred[i].argmax()
    new_pred = np.append(new_pred,max)
# print(new_pred)
acc2 = accuracy_score(y,new_pred)

print(acc2)


ran_fo = RandomForestRegressor()
ran_fo.fit(x,y)
ran_pred = ran_fo.predict(x)
r2 = r2_score(y,ran_pred)
print(r2)