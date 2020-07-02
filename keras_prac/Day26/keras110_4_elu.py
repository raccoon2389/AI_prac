import numpy as np
import matplotlib.pyplot as plt
def elu(x):
    x = np.copy(x)
    x[x<0]=0.2*(np.exp(x[x<0])-1)
    return x
    
x = np.arange(-10,5,0.1)
y = elu(x)
print(y)
print(x)
plt.figure(figsize=(12, 3))
plt.plot(x,y)
plt.grid()
plt.show()