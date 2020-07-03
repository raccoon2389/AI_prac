from keras.applications import VGG16, VGG19
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile
from keras.applications import Xception, ResNet101
from keras.optimizers import Adam
from keras.applications import ResNet101V2,ResNet152,ResNet152V2,ResNet50,ResNet50V2,InceptionV3,InceptionResNetV2

model = VGG19()
model = Xception()
model = ResNet101()
model = ResNet101V2()
model = ResNet152()
model = ResNet152V2()
model = ResNet50()
model = ResNet50V2()
model = InceptionV3()
model = InceptionResNetV2()
model = MobileNet()
model = MobileNetV2()
model = DenseNet121()
model = DenseNet169()
model = DenseNet201()
model = NASNetLarge()
model = NASNetMobile()



vgg16 = VGG16(
    weights=None, 
    include_top=False, 
    classes=10,
    input_shape=(224,224,3),
    # classifier_activation="softmax"
) 
# (None, 224, 224, 3) 
# 모델 다운 받아 와야함

vgg16.summary()

# 잘만든 모델 가져다 쓰는 거 = 전이학습
# 이미지 모델에서 준우승 한 모델

# VGG16이랑 엮기

model = Sequential()
model.add(vgg16)
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10,activation='softmax'))
model.summary()
