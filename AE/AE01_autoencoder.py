from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,Model
# Dropout 중간에 노드들을 일정비율을 지운다.
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout,Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
e_stop = EarlyStopping(monitor='loss', patience=5, mode='auto')
t_board = TensorBoard(log_dir='.\graph', histogram_freq=0,
                      write_grads=True, write_graph=True, write_images=True)
(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train[0])
print(y_train[0])

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

# plt.imshow(x_train[0],'gray')
# plt.show()

#데이터 전처리 1. 원핫인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

#데이터 전처리 2. 정규화
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1]*x_train.shape[2]).astype('float32')/255.
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1]*x_test.shape[2]).astype('float32')/255.

input_img = Input(shape=(784,))
encoded = Dense(32,activation='relu')(input_img)
decoded = Dense(784,activation='sigmoid')(encoded)

model = Model(input_img,decoded)


model.compile(optimizer='adam',
              loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, x_train, batch_size=500, epochs=30,
          validation_split=0.2)

loss, acc = model.evaluate(x_test, x_test, batch_size=100)

print(loss, acc)

pred = model.predict(x_test)



n = 10 
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(pred[i].reshape(28, 28,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()