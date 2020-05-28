import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

plt.plot(x_train)

#Sequential 형으로 작성

# 하단에 주석으로 acc와 loss 명사