# Keras50_CNN.py
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,Dense,Flatten

model = Sequential()
model.add(Conv2D(10,(5,5),strides=1,input_shape=(10,10,6)))     # filter = kernel, kernelsize
                                                                # padding(same) 커널사이즈와 같은값을 추가 해줘서 input shape와 같은 크기를 유지해준다
                                                                # d
model.add(Conv2D(10,(2,2),strides=1))

model.add(Flatten())
model.summary()