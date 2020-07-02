import numpy as np
import matplotlib.pyplot as plt

a= 0.1

def selu(a,x):
    x = np.copy(x)
    x[x<0]=a*x[x<0]
    return x
    
x = np.arange(-10,5,0.1)
y = selu(a,x)
print(y)
print(x)
plt.figure(figsize=(12, 3))
plt.plot(x,y)
plt.grid()
plt.show()