import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.layers.core import Dense
from keras.utils import np_utils
from PIL import Image

training_images = np.array(np.load('train_data.npy', allow_pickle=True))
training_labels = np.array(np.load('train_labels.npy', allow_pickle=True))
validation_images = np.array(np.load('valid_data.npy', allow_pickle=True))
validation_labels = np.array(np.load('valid_labels.npy', allow_pickle=True))
testing_images = np.array(np.load('test_data.npy', allow_pickle=True))
testing_labels = np.array(np.load('test_labels.npy', allow_pickle=True))

print(testing_images.shape)
print(training_images.shape)
print(training_labels.shape)



left = 0
right = 0
forward = 0

for i in range(len(training_labels)):
    if training_labels[i] == 0:
        forward += 1
    if training_labels[i] == 1:
        left += 1
    if training_labels[i] == 2:
        right += 1

direction = {'left': left, 'right': right, 'forward': forward}
plt.bar(direction.keys(), direction.values(), color='g')
plt.title('the number of each direction in train set')
plt.show()


img2 = Image.fromarray(training_images[0].reshape(45, 80, 3))
img3 = Image.fromarray(training_images[1].reshape(45, 80, 3))
# img2.show()
# img3.show()

#training
img = []
for i in range(training_images.shape[0]):
    each = training_images[i].reshape(45, 80, 3)
    img.append(each)
training_images = np.asarray(img)
print(training_images.shape)
#validation
img = []
for i in range(validation_images.shape[0]):
    each = validation_images[i].reshape(45, 80, 3)
    img.append(each)
validation_images = np.asarray(img)
print(validation_images.shape)
#test
img = []
for i in range(testing_images.shape[0]):
    each = testing_images[i].reshape(45, 80, 3)
    img.append(each)
testing_images = np.asarray(img)
print(testing_images.shape)



def driving():
    x_train = training_images
    x_test = testing_images
    y_train = np_utils.to_categorical(training_labels, 3)
    y_test = np_utils.to_categorical(testing_labels, 3)

    model2 = Sequential()
    model2.add(Conv2D(25, (3, 3), input_shape=(
        45, 80, 3), data_format='channels_last'))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Conv2D(50, (3, 3)))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Flatten())
    model2.add(Dense(units=100, activation='relu'))
    model2.add(Dense(units=3, activation='softmax'))
    model2.summary()

    model2.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

    model2.fit(x_train, y_train, batch_size=100, epochs=10)

    result_train = model2.evaluate(x_train, y_train)
    print('\nTrain CNN Acc:\n', result_train[1])

    result_test = model2.evaluate(x_test, y_test)
    print('\nTest CNN Acc:\n', result_test[1])

    model2.save('driving.h5')


if __name__ == '__main__':
    driving()