# Dense 와 LSTM 엮기

import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout,Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.preprocessing import StandardScaler

def split_X(seq,size):
    aaa = []
    for i in range(len(seq) - size +1 ): # len(seq) - size +1 = 몇개의 행을 갖을수 있는지 계산
        subset = seq[i: (i+size)] # 한행에 넣을 데이터 추출
        aaa.append(subset) # subset에 있는 item을 shape에 맞게 aaa 뒤에 행 추가
    return np.array(aaa)

samsung = np.load('./test_samsung/t_s.npy',allow_pickle=True)
hite = np.load('./test_samsung/t_h.npy',allow_pickle=True)

samsung = samsung.reshape(-1,)
hite = hite.reshape(-1,5)

samsung = (split_X(samsung,6))
print(samsung.shape)

# 삼성은 LSTM  하이트는 Dense로 모델링

x_sam = samsung[:,0:5]
scals = StandardScaler()
scals.fit(x_sam)
x_sam = scals.transform(x_sam)
x_sam = x_sam.reshape(-1,5,1)
y_sam = samsung[:,5]

x_hit = hite[5:510,:]
scalh = StandardScaler()
scalh.fit(x_hit)
x_hit = scalh.transform(x_hit)

input1 = Input(shape=(5,1))
x1 = LSTM(50,return_sequences=True,dropout=0.2)(input1)
x1 = LSTM(100)(x1)

input2 = Input(shape=(5,))
x2 = Dense(100)(input2)
x2 = Dense(500)(x2)

merge = concatenate([x1,x2])

output = Dense(1)(merge)

model1 = Model(inputs=[input1,input2],outputs=[output])

model1.summary()

model1.compile(optimizer='adam',loss='mse')
model1.fit([x_sam,x_hit],y_sam,batch_size=10,epochs=1000)




