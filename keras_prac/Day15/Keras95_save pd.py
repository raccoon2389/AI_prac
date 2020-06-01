import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/iris.csv",
                        index_col=None,
                        header=0,sep=',')

print(datasets)

print(datasets.head())# 앞 5개
print(datasets.tail())# 뒤 5개
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(datasets.values)# pandas -> numpy

aaa= datasets.values
x = aaa[:,0:-1]
y = aaa[:,-1]

np.save('./data/iris_x.npy', arr = x)
np.save('./data/iris_y.npy', arr = y)

print(type(aaa))

