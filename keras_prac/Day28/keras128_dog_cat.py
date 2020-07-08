from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Embedding
import numpy as np

img_dog = load_img('./data/dog_cat/dog.jpg', target_size=(224, 224))
img_yang = load_img('./data/dog_cat/yang.jpg', target_size=(224, 224))
img_cat = load_img('./data/dog_cat/cat.jpg', target_size=(224, 224))
img_suit = load_img('./data/dog_cat/suit.jpg', target_size=(224, 224))
plt.imshow(img_dog)
# plt.show()

arr_dog = img_to_array(img=img_dog)
arr_cat = img_to_array(img=img_cat)
arr_yang = img_to_array(img=img_yang)
arr_suit = img_to_array(img=img_suit)

# print(arr_dog)
# print(type(arr_dog))

arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_yang = preprocess_input(arr_yang)

arr_input = np.stack([arr_dog,arr_cat,arr_suit,arr_yang])

# print(arr_input.shape) 
#(4, 224, 224, 3)

model = VGG16()
probs = model.predict(arr_input)

# print(probs)
res=decode_predictions(probs)

print("============================================")
print(res[0])
print("============================================")
print(res[1])
print("============================================")
print(res[2])
print("============================================")
print(res[3])
print("============================================")
