import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from keras.models import Sequential, load_model
# from keras.layers import Dense,LSTM,Dropout,Conv1D,Flatten,MaxPooling1D
# from keras.callbacks import ModelCheckpoint
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold,RandomizedSearchCV,cross_val_score,cross_validate,train_test_split,KFold
from numpy.random import randint
import xgboost as xgb
from xgboost import XGBRFRegressor
from sklearn.metrics import r2_score
# m_check = ModelCheckpoint("./model/comp3/--{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)


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

# train_df = pd.DataFrame(index = train_target['id'])
# train_fe_max = train_feat.groupby(['id']).max().add_suffix('_max').iloc[:,1:]
# train_fe_min = train_feat.groupby(['id']).min().add_suffix('_min').iloc[:,1:]
# train_fe_mean = train_feat.groupby(['id']).mean().add_suffix('_mean').iloc[:,1:]
# train_fe_std = train_feat.groupby(['id']).std().add_suffix('_std').iloc[:,1:]
# train_fe_median = train_feat.groupby(['id']).median().add_suffix('_median').iloc[:,1:]
# train_fe_skew = train_feat.groupby(['id']).skew().add_suffix('_skew').iloc[:,1:]
# print(train_feat.iloc[1,1])


#################   XY만 따로 구하기    ##################

# xyzm= np.zeros((2800,4))
# col =['S1','S2','S3','S4']
# for i in range(2800):
#     idx=0
#     for p in range(4):
#         tmp = train_feat.loc[train_feat["id"]==i,col[p]].values
#         for r in range(200):
#             if tmp[r]!=0:
#                 xyzm[i,p]=r
#                 break
# np.save('./data/dacon/comp3/dist.npy',xyzm)

# xyzm= np.zeros((700,4))
# col =['S1','S2','S3','S4']
# for i in range(2800,3500,1):
#     idx=0
#     for p in range(4):
#         tmp = test_feat.loc[test_feat["id"]==i,col[p]].values
#         for r in range(200):
#             if tmp[r]!=0:
#                 xyzm[i-2800,p]=r
#                 break

# np.save('./data/dacon/comp3/dist_test.npy',xyzm)



xyzm = np.load('./data/dacon/comp3/dist.npy')
xyzm_test = np.load('./data/dacon/comp3/dist_test.npy')
d_y = train_target.loc[:,"X":"Y"]
# dx_train, dx_test, dy_train,dy_test = train_test_split(xyzm,d_y,test_size=0.2,shuffle=True,random_state=66)

# model =MultiOutputRegressor(XGBRFRegressor(learning_rate=1,n_estimators=1000))

def create_model(x_data,y_data):
    models = []
    kf = KFold(n_splits=5,shuffle=True,random_state=666)
    for train_idx,val_idx in kf.split(x_data):
        x_train = x_data[train_idx]
        y_train = y_data[train_idx]
        x_val = x_data[val_idx]
        y_val = y_data[val_idx]

        dtrain = xgb.DMatrix(x_train,label=y_train)
        dval = xgb.DMatrix(data=x_val,label=y_val)
        wlist = [(dtrain,'train'),(dval,'eval')]
        params={
            'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'seed':777,
                'gpu_id':0,
                'tree_method':'gpu_hist'
        }
        model = xgb.train(params,dtrain,num_boost_round=1000,verbose_eval=1000,evals=wlist)
        models.append(model)
    return models
models = {}
XY = pd.DataFrame(data=None,index=None)


for col in d_y.columns:
    print(col)
    models[col] = create_model(xyzm,d_y[col].values)
    print('\n\n\n\n')
    
for col in models:
    preds = []
    for model in models[col]:
        preds.append(model.predict(xgb.DMatrix(xyzm_test)))
    pred = np.mean(preds,axis=0)
    XY[col]=pred

XY.index=range(2800,3500,1)

XY.index.name="id"
XY.to_csv('./data/dacon/comp3/XY2.csv')



# model.fit(dx_train,dy_train)
# score = model.score(dx_test,dy_test)

# dist = model.predict(xyzm_test)
# XY = pd.DataFrame(dist,index=range(2800,3500,1),columns=["X","Y"])
# XY.index.name='id'
# XY.to_csv('./data/dacon/comp3/XY.csv')


#####################################################


train_df = pd.DataFrame(index = train_target['id'])
train_fe_max = train_feat.groupby(['id']).max().add_suffix('_max').iloc[:,1:]
train_fe_min = train_feat.groupby(['id']).min().add_suffix('_min').iloc[:,1:]
train_fe_mean = train_feat.groupby(['id']).mean().add_suffix('_mean').iloc[:,1:]
train_fe_std = train_feat.groupby(['id']).std().add_suffix('_std').iloc[:,1:]
train_fe_median = train_feat.groupby(['id']).median().add_suffix('_median').iloc[:,1:]
train_fe_skew = train_feat.groupby(['id']).skew().add_suffix('_skew').iloc[:,1:]
train_sum = train_feat.groupby(['id']).sum().add_suffix('_sum').iloc[:,1:]

train_df = pd.concat([train_df, train_fe_max, train_fe_min, train_fe_mean, train_fe_std, train_fe_median, train_fe_skew,train_sum], axis=1)

test_df = pd.DataFrame(index=submit['id'])
test_fe_max = test_feat.groupby(['id']).max().add_suffix('_max').iloc[:,1:]
test_fe_min = test_feat.groupby(['id']).min().add_suffix('_min').iloc[:,1:]
test_fe_mean = test_feat.groupby(['id']).mean().add_suffix('_mean').iloc[:,1:]
test_fe_std = test_feat.groupby(['id']).std().add_suffix('_std').iloc[:,1:]
test_fe_median = test_feat.groupby(['id']).median().add_suffix('_median').iloc[:,1:]
test_fe_skew = test_feat.groupby(['id']).skew().add_suffix('_skew').iloc[:,1:]
test_sum = test_feat.groupby(['id']).sum().add_suffix('_sum').iloc[:,1:]

test_df = pd.concat([test_df, test_fe_max, test_fe_min, test_fe_mean, test_fe_std, test_fe_median, test_fe_skew,test_sum], axis=1)




train_df.to_csv('./data/dacon/comp3/train_df')
test_df.to_csv('./data/dacon/comp3/test_df')


# np.save('./data/dacon/comp3/feat_pre.npy',feat)
# np.save('./data/dacon/comp3/target.npy',target)
# np.save('./data/dacon/comp3/test_pre.npy',test)




############### 데이터 불러오기 ########################
feat = pd.read_csv('./data/dacon/comp3/train_df')
# target = np.load('./data/dacon/comp3/target.npy')
test = pd.read_csv('./data/dacon/comp3/test_df')
###################################################



############### M,V트레인 #####################


print(test.shape)

mv = train_target.loc[:,"M":"V"]

def create_model(x_data,y_data):
    models = []
    kf = KFold(n_splits=5,shuffle=True,random_state=666)
    for train_idx,val_idx in kf.split(x_data):
        x_train = x_data[train_idx]
        y_train = y_data[train_idx]
        x_val = x_data[val_idx]
        y_val = y_data[val_idx]

        dtrain = xgb.DMatrix(x_train,label=y_train)
        dval = xgb.DMatrix(data=x_val,label=y_val)
        wlist = [(dtrain,'train'),(dval,'eval')]
        params={
            'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'seed':777,
                'gpu_id':0,
                'tree_method':'gpu_hist'
        }
        model = xgb.train(params,dtrain,num_boost_round=1000,verbose_eval=1000,evals=wlist)
        models.append(model)
    return models
models = {}
MV = pd.DataFrame(data=None,columns=None)


for col in mv.columns:
    print(col)
    models[col] = create_model(feat.values,mv[col].values)
    print('\n\n\n\n')
    
for col in models:
    preds = []
    for model in models[col]:
        preds.append(model.predict(xgb.DMatrix(test.values)))
    pred = np.mean(preds,axis=0)
    MV[col]=pred

MV.index=range(2800,3500,1)

MV.index.name="id"
MV.to_csv('./data/dacon/comp3/XY2.csv')


####################################################################

#####################    결과값   ###########################

MV = pd.read_csv('./data/dacon/comp3/MV.csv',index_col=0,header=0)
XY = pd.read_csv('./data/dacon/comp3/XY2.csv',index_col=0,header=0)
sub = pd.concat([XY,MV],axis=1)
sub.to_csv("./COMP3.csv")
print(sub.head())

#######################################################
###########         FFT      ##################
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