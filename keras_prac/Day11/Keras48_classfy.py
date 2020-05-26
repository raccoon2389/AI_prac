#Keras46_classfy.py
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Input
from keras.callbacks import EarlyStopping, TensorBoard
e_stop = EarlyStopping(monitor='loss', patience=50, mode='auto')
t_board = TensorBoard(log_dir='.\graph',histogram_freq=0,batch_size=1,write_graph=True, write_grads=True, write_images=True,)
import matplotlib.pyplot as plt
x= np.array(range(1,11))
y = np.array([1,0,1,0,1,0,1,0,1,0])
def binary_step(x):
    if x>0:
        return 1
    else:
        return 0

input1 = Input(shape=(1,))

dense1 = Dense(100)(input1)
desne1 = Activation(binary_step)(dense1)
dense1 = Dense(100)(dense1)
desne1 = Activation(binary_step)(dense1)
dense1 = Dense(100)(dense1)
desne1 = Activation(binary_step)(dense1)
dense1 = Dense(100)(dense1)
desne1 = Activation(binary_step)(dense1)
dense1 = Dense(1)(dense1)
desne1 = Activation(binary_step)(dense1)
dense1 = Dense(1)(dense1)
output1 = Activation(binary_step)(dense1)

model = Model(inputs=[input1],outputs=[output1])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc']) # 이진분류에서는 binary_crossentropy 하나만 쓴다.



model.fit(x,y,batch_size=1,epochs=1000,callbacks=[e_stop,t_board])

loss, acc = model.evaluate(x,y,batch_size=1)
print(f"loss : {loss}\nacc : {acc}")

y_pre = model.predict(x,batch_size=1)

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
# masked_y = np.array(y_pre>=0.5, dtype=np.int)


print(f"predict : {y_pre}")
# print(f"predict : {masked_y}")