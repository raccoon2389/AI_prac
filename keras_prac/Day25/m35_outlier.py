'''

이상치를 처리하는법
삭제하거나 NaN처리하여 보간한다
하지만 데이터에 따라 맞기도 하고 틀리기도 하다.



IQR 

Robust와 비슷
4등위로 나누어서 상위 하위 25%를 1.5를 곱한후 그 바깥 라인을 잘라낸다


'''

import numpy as np
import pandas as pd


def outliers(data_out):
    out = []
    if str(type(data_out))== str("<class 'numpy.ndarray'>"):
        for col in range(data_out.shape[1]):
            data = data_out[:,col]
            print(data)

            quartile_1, quartile_3 = np.percentile(data,[25,75])
            print("1사분위 : ",quartile_1)
            print("3사분위 : ",quartile_3)
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr*1.5)
            upper_bound = quartile_3 + (iqr*1.5)
            out_col = np.where((data>upper_bound)|(data<lower_bound))
            print(out_col)
            data = data[out_col]
            print(f"{col+1}번째 행렬의 이상치 값: ", data)
            out.append(out_col)

    elif str(type(data_out))== str("<class 'pandas.core.frame.DataFrame'>"):
        for col in data_out.columns:
            data = data_out[col].values
            quartile_1, quartile_3 = np.percentile(data,[25,75])
            print("1사분위 : ",quartile_1)
            print("3사분위 : ",quartile_3)
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr*1.5)
            upper_bound = quartile_3 + (iqr*1.5)
            data = data[np.where((data_out>upper_bound)|(data_out<lower_bound))]
            data = data[out_col]
            print(f"{col}의 이상치값: ", data)
            out.append(out_col)
    return out

a = np.array([[1,2,3,4,10000,6,7,5000,90,100],[1,2,3,4,10000,6,7,5000,90,100]])
a = a.transpose()
b = outliers(a)
print("이상치의 위치 : ",b)
a = np.array([1])
b = pd.DataFrame()
print(str(type(a))=="<class 'numpy.ndarray'>")
print(type(b))




