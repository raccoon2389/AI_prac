import numpy as np
import matplotlib.pyplot as plt
import keras.backend
import keras.losses

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5,5,0.1)
y= sigmoid(x)

print(x.shape, y.shape)

plt.plot(x,y)
plt.grid()
plt.show()