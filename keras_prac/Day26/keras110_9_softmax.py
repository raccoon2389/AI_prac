import numpy as np
import matplotlib.pyplot as plt

a= 0.1

def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x))

    
x = np.arange(-10,5,0.1)
y = softmax(x)
print(y)
print(x)

ratio = y
labels = y

plt.pie(ratio,labels=labels,shadow=True,startangle=True)
plt.show()