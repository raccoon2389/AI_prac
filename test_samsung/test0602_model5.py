#lstm 2개 구현
#Dense LSTM
import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout,Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def split_X(seq,size):
    aaa = []
    for i in range(len(seq) - size +1 ): # len(seq) - size +1 = 몇개의 행을 갖을수 있는지 계산
        subset = seq[i: (i+size)] # 한행에 넣을 데이터 추출
        aaa.append(subset) # subset에 있는 item을 shape에 맞게 aaa 뒤에 행 추가
    return np.array(aaa)

samsung = np.load('./test_samsung/t_s.npy',allow_pickle=True)
hite = np.load('./test_samsung/t_h.npy',allow_pickle=True)

samsung = samsung.reshape(-1,)

pc = PCA(n_components=2)
pc.fit(hite)
hite = pc.transform(hite)

pre_s = samsung[-5:].reshape(1,-1,1)
print(hite.shape)
pre_h = hite[-6:,:].reshape(1,-1,2)
samsung = (split_X(samsung,6))

print(pre_h.shape)

# 삼성은 LSTM  하이트는 Dense로 모델링

x_sam = samsung[:,0:5]
scals = MinMaxScaler()
scals.fit(x_sam)
x_sam = scals.transform(x_sam)
x_sam = x_sam.reshape(-1,5,1)
y_sam = samsung[:,5].reshape(-1,1)

scal = MinMaxScaler()
scal.fit(y_sam)
y_sam=scal.transform(y_sam)

x_hit = split_X(hite,6)

input1 = Input(shape=(5,1))
x1 = LSTM(50,dropout=0.2)(input1)

input2 = Input(shape=(6,2))
x2 = LSTM(10,dropout=0.2)(input2)

merge = Concatenate()([x1,x2])
output = Dense(1)(merge)

model1 = Model(inputs=[input1,input2],outputs=[output])

model1.summary()

model1.compile(optimizer='rmsprop',loss='mse')
model1.fit([x_sam,x_hit],y_sam,batch_size=30,epochs=100)


pre = model1.predict([pre_s,pre_h],batch_size=1)

print('pre : ',scal.inverse_transform(pre))
