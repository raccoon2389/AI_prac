import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wine = pd.read_csv('data/winequality-white.csv',sep=';',header=0)
count_data = wine.groupby('quality')['quality'].count()
count_data.plot()
plt.show()
print(count_data)

