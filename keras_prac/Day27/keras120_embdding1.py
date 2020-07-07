from keras.preprocessing.text import Tokenizer

text = '나는 밥을 맛있게 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])
print(token.word_index)
#{'나는': 1, '밥을': 2, '맛있게': 3, '먹었다': 4}
x = token.texts_to_sequences(([text]))
print(x)
#[[1, 2, 3, 4]]

from keras.utils import to_categorical

word_size = len(token.word_index)+1

x = to_categorical(x,num_classes=word_size)
print(x)
# [[[0. 1. 0. 0. 0.]
#   [0. 0. 1. 0. 0.]
#   [0. 0. 0. 1. 0.]
#   [0. 0. 0. 0. 1.]]]
# 단어의 개수가 늘어날때마다 데이터가 기하급수적으로 늘어난다
