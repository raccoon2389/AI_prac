from keras.layers import Dense, Embedding, LSTM, Flatten,Conv1D,MaxPool1D, BatchNormalization,Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=2000)
words = imdb.get_word_index()
# print(words)
token = Tokenizer()
token.fit_on_texts(words)
a = token.sequences_to_texts(x_train)
print(a)
# print(x_train.shape, x_test.shape)  # (8982,) (2246,)
# print(y_train.shape, y_test.shape)  # (8982,) (2246,)

# print(x_train[0]) #(25000,) (24996,)
# print(y_train[0]) #(25000,) (24996,)
#
# print(len(x_train[0])) #218
test=[]
for x in x_train[0]:
    test.append(words)
print(test)

category = np.max(y_train)+1
# print("카테고리 : ", category)  # 카테고리 :  2

# y_bunpo = np.unique(y_train)
# x_bin = np.count_nonzero(np.unique(x_train.reshape(-1,)))
# print(x_bin)

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
# print(bbb)
x_train_pd = pd.DataFrame(x_train)
# print(x_train_pd)

x_train = pad_sequences(x_train, maxlen=111, padding='pre')
x_test = pad_sequences(x_test, maxlen=111, padding='pre')

# print(x_train.shape)
# print(len(x_train[0]))

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)



model = Sequential()
model.add(Embedding(24902, 512))
# model.add(Conv1D(64,5,activation='relu'))
# model.add(MaxPool1D())
model.add(LSTM(512,dropout=0.3))
# model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))

# model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size=100,
                    epochs=10, validation_split=0.35)

acc = model.evaluate(x_test, y_test)
print('acc : ', acc)

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c='red', label='TestSet Loss')
plt.plot(y_loss, marker='.', c='blue', label='TrainSet Loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# imdb 내용확인
# word_size 전체데이터부분 변경해서 최상값 확인
# 주간과제: groupby()의 사용법 숙지할것
