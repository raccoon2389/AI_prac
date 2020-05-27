import numpy as np
from keras.utils import np_utils
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y-1
y = np_utils.to_categorical(y) # to categorical 은 무적궝ㄴ 0부터 시작을 한다.

# x = [1,2,3]
# x = x-1
# print(x)

from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1)

# aaa = OneHotEncoder(y)
# aaa.fit(y)
# y = aaa.transform(y).toarray()
# print(y)
# print(y.shape)

y = np.array([[1,2,3],[3,4,5]])

print(np.argmax(y,axis=0))