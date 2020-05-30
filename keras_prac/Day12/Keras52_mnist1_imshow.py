import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
print(y_train[0])

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

plt.imshow(x_train[1],'gray')
plt.show()