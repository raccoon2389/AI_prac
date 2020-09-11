import numpy as np
import pandas as pd 

A = [[1, 1, 1],
     [3, 2, 1],
     [2, 1, 2]]

s = [15,28,23]

r = np.linalg.inv(A)

print(r)