import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("./Data/dacon/comp1/train.csv", header=0,index_col=0)
test = pd.read_csv("./Data/dacon/comp1/test.csv", header=0,index_col=0)
submit = pd.read_csv("./Data/dacon/comp1/sample_submission.csv", header=0,index_col=0)

print('train.shape : ', train.shape) 
print('test.shape : ', test.shape)
print('sumit.shape : ', submit.shape)

# print(train.loc[train.isnull(),"650_dst"])

print(train["650_dst"].isnull().sum(),"650_dst")
test = test.interpolate()

train_dst = train.filter(regex='_dst$',axis=1)
test_dst = test.filter(regex='_dst$',axis=1)

train_dst = train_dst.interpolate()     #보간법//선형보간
test_dst = test_dst.interpolate()     #보간법//선형보간

# print(train_dst.isnull().sum(),test_dst.isnull().sum())
