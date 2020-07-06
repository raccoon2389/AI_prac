from keras.layers import Embedding, Dense, Flatten, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밌어요", '최고에요', "참 잘 만든 영화에요",
        "추천하고 싶은 영화 입니다", '한번 더 보고 싶네요', '글쎄요', '별로에요',
        '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요']

# 긍정1 부정0
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])

#토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'너무': 1, '참': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화': 10, '입니다': 11,
# '한번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17,
# '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24}
# 자주나온 단어는 인덱스를 앞으로 준다.

x = token.texts_to_sequences(docs)
print(x)
#[[1, 3], [4], [2, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [1, 23], [2, 24]]


pad_x = pad_sequences(x, padding='pre', value=0)
print(pad_x)
pad_x = pad_x.reshape(12,4,1)
word_size = len(token.word_index)+1
print(word_size)


model = Sequential()
# model.add(Embedding(25, 10, input_length=4))
model.add(LSTM(10,input_shape=(4,1)))
# model.add(Embedding(word_size,10,input_length=4))
# model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)
acc = model.evaluate(pad_x, labels)[1]
print(acc)
