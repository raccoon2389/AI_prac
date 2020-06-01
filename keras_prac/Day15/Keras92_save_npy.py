from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

print(type(iris))

x = iris.data
y = iris.target

print(type(x),type(y))

np.save('./data/iris_x.npy', arr = x)
np.save('./data/iris_y.npy', arr = y)

x_load = np.load('./data/iris_x.npy')
y_load = np.load('./data/iris_y.npy')

print(type(x_load))
print(type(y_load))