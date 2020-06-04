import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC,SVC
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import r2_score,accuracy_score
from sklearn.model_selection import train_test_split

data = load_iris()
x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, shuffle=True)
scal = MinMaxScaler()
scal.fit(x_train)
x_train = scal.transform(x_train)
x_test = scal.transform(x_test)

# y = np_utils.to_categorical(y)

lsvc = LinearSVC()
lsvc.fit(x_train,y_train)
l_pred = lsvc.predict(x_test)
score = lsvc.score(x_test,y_test)
print(score)

svc = SVC()
svc.fit(x_train,y_train)
l_pred = svc.predict(x_test)
score = svc.score(x_test,y_test)
print(score)


kne = KNeighborsClassifier(n_neighbors=1)
kne.fit(x_train,y_train)
k_pred = kne.predict(x_test)
# k_pred = np_utils.to_categorical(k_pred)
# acc1 = accuracy_score(y,k_pred)
# print(k_pred)
kne.score(x_test,y_test)
print(score)



ran_fo = RandomForestClassifier()
ran_fo.fit(x_train,y_train)
ran_pred = ran_fo.predict_proba(x)
new_pred = np.array([])
for i in range(len(ran_pred)):
    max = ran_pred[i].argmax()
    new_pred = np.append(new_pred,max)
# print(new_pred)
# acc2 = accuracy_score(y,new_pred)
score = ran_fo.score(x_test,y_test)
print(score)


ran_fo = RandomForestRegressor()
ran_fo.fit(x_train,y_train)
ran_pred = ran_fo.predict(x_test)
# r2 = r2_score(y,ran_pred)
score = ran_fo.score(x_test,y_test)

print(score)
