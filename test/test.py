import numpy as np
import pandas as pd 

df = pd.read_csv('./base.csv',index_col=0)

df.index = df['id']

df = df.iloc[:,1:]
df.index.name = 'id'
print(df)
df.to_csv('./base.csv')