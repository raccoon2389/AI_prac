import pandas as pd
import numpy as np
import glob
import pydicom as dcm
import matplotlib.pyplot as plt
import re
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,

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

plt.axis('off')
plt.imshow(img)
plt.show()

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


for file_path in glob.glob('./data/kaggle/comp1/train/*/*.dcm'):
    dataset = dcm.dcmread(file_path)
    show_dcm_info(dataset)
    break  # Comment this out to see all

