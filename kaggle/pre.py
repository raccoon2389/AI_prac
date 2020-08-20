from PIL import Image
import pandas as pd
import numpy as np
import glob
import pydicom as dcm
import matplotlib.pyplot as plt
import re
from keras.models import Sequential, load_model,Model
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D,Conv2DTranspose,Concatenate
from keras.optimizers import RMSprop,Adam
import h5py


#데이터 분석 

train = pd.read_csv('./data/kaggle/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/kaggle/comp1/test.csv', header=0, index_col=0)

# print(train)
# print(test)

folders = glob.glob('./data/kaggle/comp1/train/*')
idx = train.index.unique()
# print(folders)
# print(idx)
f_list = {}
for i,path in enumerate(folders):
    f = sorted(glob.glob(path+"/*"), key=lambda n: int(re.findall(r'\d+', n)[0]))
    f_list[idx[i]]=f
# print(f_list)

print(f_list[idx[0]][0])

r = dcm.dcmread(f_list[idx[0]][0])
img = r.pixel_array
# print(img)
img[img==-2000]=0

# plt.axis('off')
# plt.imshow(img)
# plt.show()

# plt.axis('off')
# plt.imshow(-img)  # Invert colors with -
# plt.show()


def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)
            dataset.PixelSpacing = [1, 1]
        plt.figure(figsize=(10, 10))
        plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
        plt.show()


files = glob.glob('./data/kaggle/comp1/train/*/*.dcm')

with h5py.File('./data/kaggle/comp1/train.hdf5','w') as f:
    f.create_dataset('image', (len(files), 512, 512), dtype='float32')
    image_set = f['image']

    for i,file_path in enumerate(files):
        print(f"{i}번째 \n")
        dataset = dcm.dcmread(file_path)
        # show_dcm_info(dataset)
        dataset.PixelSpacing = [1,1]
        im = dataset.pixel_array
        if(im.shape[0]==512):
            image_set[i]=im
        else:
            im = Image.fromarray(im).resize((512, 512),resample=0)
            image_set[i] = np.asfarray(im)
            print("리사이즈")
        # print(image_set.shape)
        # break  # Comment this out to see all





# 모델링

def autoencoder(hidden_laysey_size, model):
    model.add(Conv2D(hidden_laysey_size, (2, 2),
                     padding='valid', input_shape=(28, 28, 1)))
    model.add(Conv2D(hidden_laysey_size, (2, 2),
                     padding='valid'))
    model.add(Conv2D(hidden_laysey_size, (2, 2),
                     padding='valid'))
    model.add(Conv2D(hidden_laysey_size, (2, 2),
                     padding='valid'))
    model.add(Conv2DTranspose(1, (5, 5), padding='valid'))

    return model


def create_model(nodes, model):
    model.add(Conv2D(nodes*2, (2, 2), strides=1,
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(nodes*2, (2, 2), strides=1,
                     activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(nodes*4, (2, 2), padding='valid', strides=1,
                     activation='relu'))
    model.add(Conv2D(nodes*4, (2, 2), padding='valid', strides=1,
                     activation='relu'))
    model.add(Conv2D(nodes*4, (2, 2), padding='valid', strides=1,
                     activation='relu'))
    model.add(Conv2D(nodes*4, (2, 2), padding='valid', strides=1,
                     activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(nodes*8, (2, 2), padding='valid', strides=1,
                     activation='relu'))
    model.add(Conv2D(nodes*8, (2, 2), padding='valid', strides=1,
                     activation='relu'))
    model.add(Conv2D(nodes*8, (2, 2), padding='valid', strides=1,
                     activation='relu'))
    # model.add(Conv2D(nodes*8, (2, 2), padding='valid', strides=1,
    #                  activation='relu'))
    # model.add(Conv2D(nodes*8, (2, 2), padding='valid', strides=1,
    #                  activation='relu'))
    # model.add(Conv2D(nodes*8, (2, 2), padding='valid', strides=1,
    #                  activation='relu'))
    # model.add(MaxPool2D(2,2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    optimizer = RMSprop(lr=0.001, epsilon=1e-8)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['acc'])
    return model


def train():
    model = Sequential()
    model = autoencoder(32, model)
    model = create_model(32, model)
    model.fit(x=train_x, y=train_y, batch_size=30, epochs=100,
              validation_split=0.25, callbacks=[m_check])
    # pred = model.predict(train_x,batch_size=30)
    # print(pred)


def pred():
    # print(sub)
    model = load_model('./model/comp6--11--0.6531.hdf5')

    pred = model.predict(test_x, batch_size=30)
    pred = np.argmax(pred, axis=1)
    pred_df = pd.DataFrame(pred, index=range(2049, 22529), columns=["digit"])
    pred_df.index.name = "id"
    pred_df.to_csv("./comp6.csv", index=True)
    print('Done')


# train()
