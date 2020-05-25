#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. model 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(1000, input_dim =1,activation='relu' ))
model.add(Dense(100))
model.add(Dense(10))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(1,activation='relu'))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x,y, epochs=30, batch_size=1)

#4. evaluate,predict
loss, acc = model.evaluate(x,y,batch_size=1)
print("loss : ",loss,"\nacc : ",acc)