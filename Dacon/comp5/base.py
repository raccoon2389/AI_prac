import numpy as np
import pandas as pd
import glob
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential
from TaPR_pkg import etapr
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# train_list = glob.glob('./Data/dacon/comp5/train-dataset/*.csv')
# test_list = glob.glob('./Data/dacon/comp5/test-dataset/*.csv')
# print(train_list,test_list)
# ['./Data/dacon/comp5/train-dataset\\train1.csv', './Data/dacon/comp5/train-dataset\\train2.csv'] 
# ['./Data/dacon/comp5/test-dataset\\test1.csv', './Data/dacon/comp5/test-dataset\\test2.csv']


# def dataframe_from_csv(target):
#     return pd.read_csv(target,index_col=0).rename(columns=lambda x: x.strip())


# def dataframe_from_csvs(targets):
#     return pd.concat([dataframe_from_csv(x) for x in targets])

# train_data = dataframe_from_csvs(train_list)
# test_data = dataframe_from_csvs(test_list)

# train_data.to_csv('./Data/dacon/comp5/train.csv')
# test_data.to_csv('./Data/dacon/comp5/test.csv')

train_data = pd.read_csv('./Data/dacon/comp5/train.csv',header=0,index_col=0)
test_data = pd.read_csv('./Data/dacon/comp5/test.csv',header=0,index_col=0)

# print(train_data,test_data)

# [550800 rows x 63 columns]
# [444600 rows x 63 columns]

scal = StandardScaler()
scal.fit(train_data)
train_data = scal.transform(train_data)
test_data = scal.transform(test_data)

x = train_data[1:-4]
y = train_data[-4:-1]
print(x)
print(y)
x_data = TimeseriesGenerator(x,x,length='look_back',sampling_rate=1,stride=1,batch_size=90)
y_data = TimeseriesGenerator(
    y, y, length='look_back', sampling_rate=1, stride=1, batch_size=90)
