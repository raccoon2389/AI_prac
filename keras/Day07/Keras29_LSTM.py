import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense,LSTM

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])
y2 = np.array([[4,5,6,7]])
y3 = np.array([[4],[5],[6],[7]])
y4 = np.zeros((2,4,2))
print("shape of x : ", x.shape)
print("shape of y : ", y.shape)
print("shape of y2 : ", y2.shape)
print("shape of y3 : ", y3.shape)

print(y4)
x = x.reshape(4,3,1)
print("shape of x : ", x.shape)


x = x.reshape(x.shape[1],x.shape[0],1)
print("shape of x : ", x.shape)
