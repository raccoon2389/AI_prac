import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datestrs = ['6/1/2020','6/3/2020','6/4/2020','6/8/2020','6/10/2020']
dates = pd.to_datetime(datestrs)
print(dates)
print('=========================')

ts = pd.Series([1,np.nan,np.nan,8,10],index=dates)
print(ts)

ts_intp_linear = ts.interpolate()
print(ts_intp_linear)