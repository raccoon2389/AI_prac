#Keras46_classfy.py
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, TensorBoard
from keras import activations
e_stop = EarlyStopping(monitor='loss', patience=50, mode='auto')
t_board = TensorBoard(log_dir='.\graph',histogram_freq=0,write_graph=True, write_grads=True, write_images=True,)

x= np.array(range(1,11))
y = np.array([1,0,1,0,1,0,1,0,1,0])
def binary_step(x):
    return np.array(x>0,dtype=np.int)

# print(binary_step(x))



model = Sequential()
model.add(Dense(60,activation='relu',input_dim=1))
model.add(Dense(1,activation='sigmoid'))



model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc']) # 이진분류에서는 binary_crossentropy 하나만 쓴다.



model.fit(x,y,batch_size=1,epochs=1000,callbacks=[e_stop,t_board])

y_pre = model.predict(x,batch_size=1)

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
masked_y = np.array(y_pre>=0.5, dtype=np.int)

loss, acc = model.evaluate(x,y,batch_size=1)

print(f"loss : {loss}\nacc : {acc}")

print(f"predict : {y_pre}")
# print(f"predict : {masked_y}")
