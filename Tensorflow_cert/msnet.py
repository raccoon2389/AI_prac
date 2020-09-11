from PIL import Image
from autocrop import Cropper
cropper = Cropper()
from tensorflow.data.dataset import
from keras.callbacks import Chec

for i in range(150):
    try:
        cropped_array = cropper.crop('ms/IU/'+str(i)+'.jpg')
        # print(cropped_array)
        cropped_image = Image.fromarray(cropped_array)
        cropped_image.save('ms/IU_cropped/'+str(i)+'.jpg')
    except AttributeError:
        print(i)
