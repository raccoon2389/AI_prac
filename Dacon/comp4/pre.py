import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dat = pd.read_csv('./data/dacon/comp4/201901-202003.csv')


for col in dat.columns:
    nu = dat[col].isnull().sum()
    print(f"{col}의 결측치는 {nu}개 입니다.")

