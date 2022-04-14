import keras
import numpy as np
import cv2

# load
model = keras.models.load_model('driving.h5')

sub_dirs = 'D:\dataset_driving'
classcatalogue = ['forward','left','right']
image = cv2.imread(sub_dirs + '\\' + 'image1.png')
img = cv2.resize(image, (45, 80))
img = (img.reshape(1, 45, 80, 3)).astype('float32')
predict = np.argmax(model.predict(img), axis=-1)

print(classcatalogue[int(predict)])

