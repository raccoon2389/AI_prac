import numpy as np
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import r2_score,accuracy_score


data = load_boston()
x = data.data
y = data.target

kne = KNeighborsRegressor(n_neighbors=3)
kne.fit(x,y)
k_pred = kne.predict(x)
# k_pred = np_utils.to_categorical(k_pred)
acc1 = r2_score(y,k_pred)
# print(k_pred)
print(acc1)



# ran_fo = RandomForestClassifier()
# ran_fo.fit(x,y)
# ran_pred = ran_fo.predict_proba(x)
# # new_pred = np.array([])
# # for i in range(len(ran_pred)):
# #     max = ran_pred[i].argmax()
# #     new_pred = np.append(new_pred,max)
# # # print(new_pred)
# acc2 = r2_score(y,ran_pred)

# print(acc2)


ran_fo = RandomForestRegressor()
ran_fo.fit(x,y)
ran_pred = ran_fo.predict(x)
r2 = r2_score(y,ran_pred)
print(r2)