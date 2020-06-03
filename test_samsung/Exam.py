# 삼성전자 6/3 시가 
# csv파일 자체 건들지 말것
# 앙상블 모델 사용
# 6시 이전에 메일제목 "홍석규, [0622시험], ~~원""  
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import LSTM,Dense,Input
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# 자르기 함수
def split_X(seq,size):
    aaa = []
    for i in range(len(seq) - size +1 ): # len(seq) - size +1 = 몇개의 행을 갖을수 있는지 계산
        subset = seq[i: (i+size)] # 한행에 넣을 데이터 추출
        aaa.append(subset) # subset에 있는 item을 shape에 맞게 aaa 뒤에 행 추가
    return np.array(aaa)


#콜백 함수
e_stop = EarlyStopping(monitor='val_loss',patience=20,mode='auto')
m_point = ModelCheckpoint(filepath=".\model\Exam--{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)


#csv 불러온후 npy로 저장
'''
samsung = pd.read_csv('./data/Samsung.csv',index_col='일자' , header=0 , sep=',',encoding='euc-kr')
# print(samsung.head())
hite = pd.read_csv('./data/Hite.csv',index_col='일자' , header=0 , sep=',',encoding='euc-kr')



# print(hite.head())

# print(samsung)

samsung = samsung.loc[:,'시가'].sort_index(axis=0,ascending=True).dropna().str.replace(',','',regex=True).values.astype('int')


hite = hite.loc[:,'시가':].dropna().sort_index(axis=0,ascending=True).replace(',','',regex=True).values.astype('int')

hite = np.append(hite,[[39000,0,0,0,0]],axis=0)

np.save('./data/sam.npy',arr=samsung)
np.save('./data/hite.npy',arr=hite)
'''
#npy 불러오기
samsung = np.load('./data/sam.npy')
hite = np.load('./data/hite.npy')

#하이트 오늘 결측치 어제수치를 그대로 붙히기

hite[-1,1:] = hite[-2,1:]

#하이트 PCA를 이용한 행 축소

# hite_PCA = PCA(n_components=1)
# hite_PCA.fit(hite)
# hite = hite_PCA.transform(hite)


# print(samsung)
# print(hite)

#############       데이터 사이즈       ###########


size= 50


############                            #############

########        하이트 전처리         ######

#하이트 시가와 거래량만 끌어옴
hite = hite[:,(0,-1)]

print(hite.shape)

#하이트 스케일링
hite_scal = MinMaxScaler()
hite_scal.fit(hite)
hite = hite_scal.transform(hite)

# 데이터 스플릿
hite_set = split_X(hite,size)
print(hite_set.shape)

#x y 셋 슬라이싱
hite_x = hite_set[:,:-1]
# hite_y = hite_set[:,-1] #안씀




#################################################




###################         삼성 전처리         ##################

#삼성 스케일링
samsung = samsung.reshape(-1,1)
sam_scal = MinMaxScaler()
sam_scal.fit(samsung)
sam_scaled = sam_scal.transform(samsung)

# 데이터 스플릿

sam_set =split_X(sam_scaled,size)

#x 셋 슬라이싱

sam_x = sam_set[:,:-1]      #1~19일치
sam_y = sam_set[:,-1]       #20번째 날







#############################################



########            차원 맞춰주기           #######
# print(sam_x.shape,sam_y.shape,hite_x.shape,hite_y.shape) 
# ---------->>>>>(460, 49) (460,) (460, 49, 2) (460, 2)

sam_x = sam_x.reshape(-1,size-1,1)
sam_y = sam_y.reshape(-1,1)




#################################################

########## 내일 예상을 위한 데이터 ###########

pred_sam_x = sam_set[-1,1:]
print(pred_sam_x.shape)

pred_sam_x = pred_sam_x.reshape(1,-1,1)
#ValueError: Error when checking input: expected input_1 to have shape (49, 1) but got array with shape (2450, 1)

pred_hite_x = hite_set[-1,1:]
print(pred_hite_x.shape)

pred_hite_x = pred_hite_x.reshape(1,-1,2)

############################################





# print(sam_x.shape) #490,size-1,1


# print(sam_x)
# print(sam_y)
##################################################
#######

####
sam_train_x, sam_test_x, \
    sam_train_y, sam_test_y, \
        hite_train_x, hite_test_x \
            = train_test_split(sam_x,sam_y,hite_x, test_size= 0.2, shuffle= False)

print(sam_train_x.shape, sam_test_x.shape, 
    sam_train_y.shape, sam_test_y.shape, 
        hite_train_x.shape, hite_test_x.shape)

print(sam_test_x[0],hite_train_x[0],sam_train_y[0])

# 예측 모델링
input_sam = Input(shape=(size-1,1))
input_hite = Input(shape=(size-1,2))

lstm_s = LSTM(60,activation='tanh',dropout=0.2)(input_sam)
lstm_h = LSTM(60,activation='tanh',dropout=0.2)(input_hite)

merge = concatenate([lstm_s,lstm_h])

output1 = Dense(100,activation='relu')(merge)
output1 = Dense(100,activation='relu')(output1)
output1 = Dense(100,activation='relu')(output1)
output1 = Dense(1)(output1)

model= Model(inputs=[input_sam,input_hite],outputs=[output1])

model.summary()

model.compile(optimizer='adam',loss='mse',metrics=['acc'])
hist = model.fit([sam_train_x,hite_train_x],sam_train_y,batch_size=5,epochs=100,validation_split=0.25,callbacks=[e_stop,m_point])
model.save('./model/!EXAM.h5')

#평가
loss ,acc = model.evaluate([sam_test_x,hite_test_x],sam_test_y,batch_size=1)

y_predict = model.predict([sam_test_x,hite_test_x],batch_size=1)

r2 = r2_score(sam_test_y, y_predict)


#내일 주가 예측
pred_next_sam = model.predict([pred_sam_x,pred_hite_x],batch_size=1)
pred_next_sam = sam_scal.inverse_transform(pred_next_sam)
print(f"loss : {loss}\nacc : {acc}\nR2 : {r2}\nnext : {pred_next_sam}")

#그래프 그려보기
plt.plot(hist.history['loss'])
plt.show()

#loss : 0.03133463976942237
# acc : 0.021739130839705467
# R2 : 0.3083589001574576
# next : [[47915.16]]