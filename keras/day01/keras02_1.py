#p.37

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.arange(1,21)
y_train = x_train*2
x_test = np.arange(101,121)
y_test = x_test*2

model = Sequential()
'''model.add(Dense(5000,input_dim=1,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='relu'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train,y_train,epochs=1000,batch_size=5,validation_data=(x_test,y_test))
loss,acc = model.evaluate(x_test,y_test,batch_size=1)

print("loss : ", loss)
print("acc = ",acc)

output = model.predict(x_test)
print("result : \n", output)'''
model.summary()