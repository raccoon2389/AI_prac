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
from sklearn.model_selection import KFold,RandomizedSearchCV,cross_val_score,cross_validate
from numpy.random import randint
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

train_df = pd.DataFrame(index = train_target['id'])
train_fe_max = train_feat.groupby(['id']).max().add_suffix('_max').iloc[:,1:]
train_fe_min = train_feat.groupby(['id']).min().add_suffix('_min').iloc[:,1:]
train_fe_mean = train_feat.groupby(['id']).mean().add_suffix('_mean').iloc[:,1:]
train_fe_std = train_feat.groupby(['id']).std().add_suffix('_std').iloc[:,1:]
train_fe_median = train_feat.groupby(['id']).median().add_suffix('_median').iloc[:,1:]
train_fe_skew = train_feat.groupby(['id']).skew().add_suffix('_skew').iloc[:,1:]
train_df = pd.concat([train_df, train_fe_max, train_fe_min, train_fe_mean, train_fe_std, train_fe_median, train_fe_skew], axis=1)

test_df = pd.DataFrame(index=submit['id'])
test_fe_max = test_feat.groupby(['id']).max().add_suffix('_max').iloc[:,1:]
test_fe_min = test_feat.groupby(['id']).min().add_suffix('_min').iloc[:,1:]
test_fe_mean = test_feat.groupby(['id']).mean().add_suffix('_mean').iloc[:,1:]
test_fe_std = test_feat.groupby(['id']).std().add_suffix('_std').iloc[:,1:]
test_fe_median = test_feat.groupby(['id']).median().add_suffix('_median').iloc[:,1:]
test_fe_skew = test_feat.groupby(['id']).skew().add_suffix('_skew').iloc[:,1:]
test_df = pd.concat([test_df, test_fe_max, test_fe_min, test_fe_mean, test_fe_std, test_fe_median, test_fe_skew], axis=1)


feat = train_df.values
test = test_df.values


np.save('./data/dacon/comp3/feat_pre.npy',feat)
# np.save('./data/dacon/comp3/target.npy',target)
np.save('./data/dacon/comp3/test_pre.npy',test)


'''

############### 데이터 불러오기 ########################
feat = np.load('./data/dacon/comp3/feat_pre.npy')
target = np.load('./data/dacon/comp3/target.npy')
test = np.load('./data/dacon/comp3/test_pre.npy')
###################################################



############### 1번 트레인 #####################

####################################################################

#####################  ML ###########################


kf = KFold()
param = [
    {"ranfo__n_estimators": randint(2,100)},
    {"ranfo__":}
]
piii = Pipeline([("scal",MinMaxScaler()),("ranfo",RandomForestRegressor())])
model = RandomizedSearchCV(piii,param_distributions=param)

model.fit(feat,target)
y = model.predict(test)
print(y)

df = pd.DataFrame(y,index=range(2800,3500,1),columns=["X","Y","M","V"])

df.to_csv('./comp3.csv')

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