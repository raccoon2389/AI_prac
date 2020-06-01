#95번을 불러와서 모델을 완성하시오

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

x = np.load('./data/iris_x.npy')
y = np.load('./data/iris_y.npy')

y = np_utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle= True, test_size=0.2)


model = Sequential()

model.add(Dense(100,activation='relu', input_shape=(4,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['mse'])
model.fit(x_train,y_train,batch_size=1,epochs=100,validation_split=0.25)

loss, mse = model.evaluate(x_test,y_test,batch_size=1)

print(loss,mse)