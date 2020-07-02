import numpy as np
import matplotlib.pyplot as plt
def relu(x):
    x1 = np.copy(x)
    x1[x1<0]=0
    return x1
x = np.arange(-5,5,0.1)
print(x)
y = relu(x)
print(x)

plt.plot(x,y)
plt.grid()
plt.show()