# Keras49_Classify.py
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import to_categorical
e_stop = EarlyStopping(monitor='loss', patience=50, mode='auto')
t_board = TensorBoard(log_dir='.\graph',histogram_freq=0,batch_size=1,write_graph=True, write_grads=True, write_images=True,)
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
x= np.array(range(1,11))
y = np.array([1,2,3,4,5,1,2,3,4,5])

y = to_categorical(y)


model = Sequential()
model.add(Dense(100,activation='relu',input_dim=1))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(6,activation='sigmoid'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

model.fit(x,y,batch_size=1,epochs=100,callbacks=[e_stop,t_board])

loss, acc = model.evaluate(x,y,batch_size=1)
print(f"loss : {loss}\nacc : {acc}")
y_pre = model.predict(x,batch_size=1)

print(f"x : {x}\npredict : {y_pre}")