import numpy as np
from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Dense,LSTM, Conv2D,Flatten,MaxPooling2D,Dropout
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

plt.imshow(x_train[1])
plt.show()

