#p.37

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.arange(1,11)
y_train = np.arange(1,11)
x_test = np.arange(101,111)
y_test = np.arange(101,111)

model = Sequential()


model.add(Dense(5,input_dim=1,activation='relu'))
model.add(Dense(3))
model.add(Dense(1,activation='relu'))

model.summary()
'''
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_train,y_train))
loss,acc = model.evaluate(x_test,y_test,batch_size=1)

print("loss : ", loss)
print("acc = ",acc)

output = model.predict(x_test)
print("result : \n", output)
'''
