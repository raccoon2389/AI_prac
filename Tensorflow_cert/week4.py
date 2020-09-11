# from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.applications.inception_v3 import InceptionV3



local_weight_file = '/tmp/inceptionpre_'
pre_trained_model = InceptionV3(input_shape = (299,299,3))
# for layer in pre_trained_model.layers:
#     layer. = False

# pre_trained_model.summary()

# last_layer = pre_trained_model.get_layer('mixed7')

# from tensorflow

# x = layers.Flatten((last_output))
# x = layers.Dense()
# x = layers.Dense()

# model = Model(pre_trained_model.input,x)

# train_datagen = Generator
