import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, MaxPool2D,Flatten, Dropout #Dropout 중간에 노드들을 일정비율을 지운다.
from keras.callbacks import EarlyStopping,ModelCheckpoint

m_check = ModelCheckpoint(filepath=".\keras_prac\model\{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)
e_stop = EarlyStopping(monitor='loss',patience=5,mode='auto')
(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train[0])
print(y_train[0])

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

# plt.imshow(x_train[0],'gray')
# plt.show()
np.save('./data/mnist_train_x.npy', arr = x_train)
np.save('./data/mnist_train_y.npy', arr = y_train)
np.save('./data/mnist_test_x.npy', arr = x_test)
np.save('./data/mnist_test_y.npy', arr = y_test)

#데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


print(y_train.shape)

#데이터 전처리 2. 정규화
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1).astype('float32')/255.