import pandas as pd
import numpy as np

samsung = pd.read_csv('./data/Samsung.csv',index_col=0 , header=0 , sep=',',encoding='CP949')
hite = pd.read_csv('./data/Hite.csv',index_col=0 , header=0 , sep=',',encoding='CP949')

#none 제거1
samsung=samsung.dropna(axis=0)
hite = hite.fillna(method='bfill')
hite = hite.dropna(axis=0)
# print(hite.loc[hite.loc[:,"시가1"].isnull()].dropna())
# print(hite.head())
#none 제거2
# hite = hite[0:509]
# hite.iloc[0,1:5]=[10,20,30,40]
# hite.loc['2020-06-02','고가':'거래량']=[100,200,300,400]

for i in range(len(samsung.index)):
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',',''))

for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        hite.iloc[i,j] = int(hite.iloc[i,j].replace(',',''))

samsung = samsung.values
hite = hite.values
print(hite)

np.save('./test_samsung/t_s.npy',arr=samsung)
np.save('./test_samsung/t_h.npy',arr=hite)
