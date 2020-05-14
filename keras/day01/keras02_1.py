#p.37

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.arange(1,21)
y_train = x_train*2
x_test = np.arange(101,121)
y_test = x_test*2

model = Sequential()
model.add(Dense(3000,input_dim=1,activation='relu')) #modle.add(레이어 추가)   (Dense(노드수)) relu활성화 함수
model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='relu'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #mse=mean_square_error
model.summary()

model.fit(x_train,y_train,epochs=100,batch_size=2,validation_data=(x_test,y_test)) #batch_size defualt = 32 epoch=반복횟수 
                                                                                #validation_data = over-fitting을 피하기 위해 train set중 일부를 따로 빼서 준비해야하지만 여기서는 test로 대체 
loss,acc = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ", loss)
print("acc = ",acc)

output = model.predict(x_test)
print("result : \n", output)