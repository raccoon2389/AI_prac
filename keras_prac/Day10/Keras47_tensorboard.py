import os,sys
import numpy as np
from keras.models import Sequential# ,load_model
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping,TensorBoard
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Day09.Keras39_split import split_X
from sklearn.model_selection import train_test_split

#1 데이터
a = np.array(range(1,101))
size = 5                # time_steps= 5

#데이터 자르기
dataset = split_X(a,size)
x_set = dataset[:,:-1]
y_set = dataset[:,-1]

print(x_set.shape)

#train, test 분리하기 (8:2)

# x_train = x_set[:80]
# y_train = y_set[:80]
# print(x_train.shape)


x_train,x_test, y_train,y_test = train_test_split(x_set,y_set, test_size=0.2, shuffle=True)
# x_train ,x_test = x_train.reshape(-1,x_train.shape[1],1), x_test.reshape(-1,x_test.shape[1],1)

# 마지막 6행을 predict 만들기

x_predict = x_set[-6:]

# x_predict = x_predict.reshape(-1,4,1)


# model = load_model("./keras_prac/model/save_keras_45.h5")
model = Sequential()
model.add(Dense(10,input_shape=(4,)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1, name= 'new'))

model.summary()

model.compile(optimizer='adam', loss = 'mse')

e_stop = EarlyStopping(monitor='loss',patience=40,mode='auto')
t_board = TensorBoard(log_dir='.\graph', histogram_freq=0,batch_size=1,write_graph=True, write_images=True)
model.fit(x_train,y_train,batch_size=1,epochs=10000,validation_split=0.2,callbacks=[e_stop,t_board])

mse = model.evaluate(x_train, y_train, batch_size=1)

print("MSE : ",mse)
y_predict = model.predict(x_predict,batch_size=1)
print(x_predict,y_predict)