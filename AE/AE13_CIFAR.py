from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
# Dropout 중간에 노드들을 일정비율을 지운다.
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout, Input, Conv2DTranspose, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import random


def autoencoder(hidden_laysey_size):
    model = Sequential()
    # model.add(Dense(units=hidden_laysey_size,
    # input_shape=(784,), activation='relu'))
    model.add(Conv2D(hidden_laysey_size, (2, 2),
                     padding='valid', input_shape=(height, width, channel)))
    model.add(Conv2D(hidden_laysey_size, (2, 2),
                     padding='valid'))
    model.add(Conv2D(hidden_laysey_size, (2, 2),
                     padding='valid', input_shape=(height, width, channel)))
    model.add(Conv2D(hidden_laysey_size, (2, 2),
                     padding='valid'))
    model.add(Conv2DTranspose(3, (5, 5), padding='valid'))
   

    return model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#데이터 전처리 1. 원핫인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

height,width,channel = x_train.shape[2], x_train.shape[1],x_train.shape[3]


#데이터 전처리 2. 정규화
x_train = x_train.reshape(-1, height, width, channel)/255.
#     x_train.shape[0], x_train.shape[1]*x_train.shape[2]).astype('float32')/255.
x_test = x_test.reshape(-1, height, width, channel)/255.
#     x_test.shape[0], x_test.shape[1]*x_test.shape[2]).astype('float32')/255.

# x_train_noise = x_train + np.random.normal(0, 0.2, size=x_train.shape)
# x_test_noise = x_test + np.random.normal(0, 0.2, size=x_test.shape)
# x_train_noise = np.clip(x_train_noise, 0, 1)
# x_test_noise = np.clip(x_test_noise, 0, 1)

model = autoencoder(hidden_laysey_size=32)
model.summary()
# model.compile(optimizer='adam', loss='mse', metrics=['acc'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train, x_train, epochs=10)


fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)
      ) = plt.subplots(3, 5, figsize=(20, 7))

output = model.predict(x_test)


random_images = random.sample(range(output.shape[0]), 5)

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(height, width,channel))
    if i == 0:
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test[random_images[i]].reshape(height, width,channel))
    if i == 0:
        ax.set_ylabel("NOISE", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(height, width,channel))
    if i == 0:
        ax.set_ylabel("OUTPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
