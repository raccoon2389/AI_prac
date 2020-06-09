import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense,LSTM,Dropout,Conv1D,Flatten,MaxPooling1D
from keras.callbacks import ModelCheckpoint
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

m_check = ModelCheckpoint("./model/comp3/--{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)


def split_X(seq,size):
    scal = MinMaxScaler()
    scal.fit(seq)
    seq = scal.transform(seq)
    aaa = np.zeros((len(seq) - size+1,size,4),dtype=np.float64)
    # print(seq.shape)
    idx = 0
    for i in range(0,len(seq) - size +1): # len(seq) - size +1 = 몇개의 행을 갖을수 있는지 계산
        subset = seq[i: (i+size),:] # 한행에 넣을 데이터 추출
        aaa[idx]= subset # subset에 있는 item을 shape에 맞게 aaa 뒤에 행 추가
        idx += 1

    # print(aaa[0])
    return (aaa, seq[len(seq)-size:] ,scal)


def kaeri_metric(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''
    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)


### E1과 E2는 아래에 정의됨 ###

def E1(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)


def E2(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    
    
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))


'''
train_feat = pd.read_csv('./data/dacon/comp3/train_features.csv',sep=',',header=0)
train_target = pd.read_csv('./data/dacon/comp3/train_target.csv',sep=',',header=0)
test_feat= pd.read_csv('./data/dacon/comp3/test_features.csv',sep=',',header=0)
submit = pd.read_csv('./data/dacon/comp3/sample_submission.csv',sep=',',header=0)

######데이터 분석########
# print(test_feat.head())
# train_feat.loc[train_feat.loc[:,"id"]==0,"S1"].plot()
# plt.show()
#########################

# 결측치 측정

# print(train_feat.isnull().sum()) ## 0
# print(train_target.isnull().sum()) ## 0

# print(train_feat["id"].max())
########################

feat = np.zeros((train_feat["id"].max()+1,train_feat.loc[train_feat.loc[:,"id"]==0].shape[0],4),dtype=np.float64)
target = np.zeros((train_target["id"].max()+1,4),dtype=np.float64)
test= np.zeros((700,375,4),dtype=np.float64)

# for i in range(train_feat["id"].max()+1):
#     feat[i] = train_feat.loc[train_feat.loc[:,"id"]==i,'S1':].values
#     target[i] = train_target.loc[train_target.loc[:,"id"]==i,"X":].values


for i in range(700):
    test[i] = test_feat.loc[test_feat.loc[:,"id"]==i+2800,'S1':].values



print(test.shape)
# np.save('./data/dacon/comp3/feat.npy',feat)
# np.save('./data/dacon/comp3/target.npy',target)
np.save('./data/dacon/comp3/test.npy',test)

'''


############### 데이터 불러오기 ########################
feat = np.load('./data/dacon/comp3/feat.npy')
target = np.load('./data/dacon/comp3/target.npy')
test = np.load('./data/dacon/comp3/test.npy')
###################################################



############### 1번 트레인 #####################
'''
# train2 = np.zeros((2800,10,4))
test2 = np.zeros((700,10,4))
scal = MinMaxScaler()
tmp = feat.reshape(2800,-1)
scal.fit(tmp)
print(test)
test = test.reshape(700,375*4)
test = scal.transform(test)
test = test.reshape(700,-1,4)

# for i in range(2800):
#     tmp , pred, scal = split_X(feat[i],50)

#     print(tmp.shape)

#     tmp = tmp.reshape(-1,50,4)
#     pred = pred.reshape(1,50,4)
#     pred = pred[:,10:,:]
#     x = tmp[:,:-10,:]
#     y= tmp[:,-10:,:].reshape(-1,40)

#     print(y.shape)
#     # y = scal.inverse_transform(y)

#     model = Sequential()
#     model.add(Conv1D(40,6,input_shape=(40,4)))
#     model.add(Flatten())
#     model.add(Dense(100,activation='relu'))
#     model.add(Dense(40))
#     model.compile(optimizer='rmsprop',loss='mse',metrics=['mse'])
#     model.fit(x,y,batch_size=20,epochs=30,validation_split=0.25)
#     y = model.predict(pred)
#     y = y.reshape(-1,10,4)
#     train2[i] = y
#     np.save('./data/dacon/comp3/train2.npy',train2)

for i in range(700):
    tmp , pred, scal = split_X(test[i],50)

    # print(tmp.shape)

    tmp = tmp.reshape(-1,50,4)
    pred = pred.reshape(1,50,4)
    pred = pred[:,10:,:]
    x = tmp[:,:-10,:]
    y= tmp[:,-10:,:].reshape(-1,40)

    # print(y.shape)
    # y = scal.inverse_transform(y)

    model = Sequential()
    model.add(Conv1D(40,6,input_shape=(40,4)))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(40))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mse'])
    model.fit(x,y,batch_size=20,epochs=30,validation_split=0.25)
    y = model.predict(pred)
    y = y.reshape(-1,10,4)
    test2[i] = y
    np.save('./data/dacon/comp3/test2.npy',test2)
'''
##############################################################


##################### 2번째 train ###################

train = np.load('./data/dacon/comp3/train2.npy')
test = np.load('./data/dacon/comp3/test2.npy')

# train = train.reshape(-1,40)
print(test.shape)

# model = Sequential()

# model.add(LSTM(200,input_shape=(10,4)))
# model.add(Dropout(0.4))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(4))

# model.compile(optimizer='rmsprop',loss='mse')
# model.fit(train,target,batch_size=20,epochs=150,validation_split=0.25,callbacks=[m_check])


model = load_model('./model/comp3/--117--24959.8967.hdf5')

y = model.predict(test)
print(y)


df = pd.DataFrame(y,index=range(2800,3500,1),columns=["X","Y","M","V"])

df.to_csv('./comp3.csv')

####################################################################

#####################  ML ###########################
'''

for i in range(2800):
    tmp = feat[i,:,:]
    pipe = Pipeline([("scal",MinMaxScaler()),("ran",RandomForestRegressor())])
    # ranfo = MultiOutputRegressor(pipe())
    pipe.fit(tmp)
    y= pipe.predict(tmp)
    print(y,i)
'''
####################################################


# FFT
'''
# feat = split_X(feat,10)
print(feat.shape)

# plt.plot(feat[0])
fft = np.zeros((2800,375,4))
for q in range(2800):
    for i in range(4):
        tmp = feat[q,:,i]
        
        fft[q,:,i] = np.fft.fft(tmp,norm='ortho').imag

plt.plot(fft[0,:,0])

plt.show()
'''


# for i in range(4):
#     scal = MinMaxScaler()
#     scal.fit(feat[:,:,i].reshape(-1,1))
#     feat[:,:,i] = scal.transform(feat[:,:,i].reshape(-1,1)).reshape(2800,375)



'''
print(feat)

model = Sequential()
model.add(LSTM(50,activation='tanh',dropout=0.25,input_shape=(feat.shape[1],feat.shape[2])))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(4))

model.compile(optimizer='rmsprop',loss='mse',metrics=['mse'])
model.fit(feat,target,batch_size=60,epochs=100,validation_split=0.25)
'''