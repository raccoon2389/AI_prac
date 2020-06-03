#None 제거
import pandas as pd
import numpy as np

samsung = pd.read_csv('./data/Samsung.csv',index_col='일자' , header=0 , sep=',',encoding='cp949')
hite = pd.read_csv('./data/Hite.csv',index_col='일자' , header=0 , sep=',',encoding='cp949')


hite = hite.fillna(method='bfill')
hite = hite.dropna(axis=0)

hite = hite[0:509]
hite.iloc[0,1:5]=[10,20,30,40]
hite.loc['2020-06-02','고가':'거래량']=[10,20,30,40]
