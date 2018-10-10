# imports
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import json
import cv2

"""
num_imgs = 10

img_size = 32
num_objects = 2

imgs = np.zeros((num_imgs, img_size, img_size, 4), dtype=np.uint8)  # format: BGRA
shapes = np.zeros((num_imgs, num_objects), dtype=int)
num_shapes = 13
num_text = 36
shape_labels = ['circle', 'semicircle', 'quartercircle', 'triangle', 'square', 'rectangle', 'trapezoid', 'pentagon', 'hexagon', 'heptagon', 'octagon', 'star', 'cross']
text_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = np.zeros((num_imgs, num_objects), dtype=int)
num_colors = 3
color_labels = ['r', 'g', 'b']
"""


num_imgs = 100

img_size = 32
min_object_size = 4
max_object_size = 16
num_objects = 2

bboxes = np.zeros((num_imgs, num_objects, 4))
imgs = np.zeros((num_imgs, img_size, img_size, 4), dtype=np.uint8)  # format: BGRA
shapes = np.zeros((num_imgs, num_objects), dtype=int)
num_shapes = 3
shape_labels = ['rectangle', 'circle', 'triangle']
shape_label = ['circle', 'semicircle', 'quartercircle', 'triangle', 'square', 'rectangle', 'trapezoid', 'pentagon', 'hexagon', 'heptagon', 'octagon', 'star', 'cross']
colors = np.zeros((num_imgs, num_objects), dtype=int)
num_colors = 3
color_labels = ['r', 'g', 'b']


def create_model():
    # Activate GPU for this, otherwise the convnet will take forever to train with Theano.

    # TODO: Make one run with very deep network (~10 layers).
    filter_size = 3
    pool_size = 2

    # TODO: Maybe remove pooling bc it takes away the spatial information.

    model = Sequential([
        Convolution2D(32, 6, 6, input_shape=(32, 32, 3), dim_ordering='tf', activation='relu'), 
        MaxPooling2D(pool_size=(pool_size, pool_size)), 
        Convolution2D(64, filter_size, filter_size, dim_ordering='tf', activation='relu'), 
        MaxPooling2D(pool_size=(pool_size, pool_size)), 
        Convolution2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'), 
        # #         MaxPooling2D(pool_size=(pool_size, pool_size)), 
        Convolution2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'), 
        # #         MaxPooling2D(pool_size=(pool_size, pool_size)), 
        Flatten(), 
        Dropout(0.4), 
        Dense(256, activation='relu'), 
        Dropout(0.4), 
        Dense(20)
    ])

    return model

l = []

def getImgs(imgs_path, h, w):
    imgs = []
    for filename in glob.glob(imgs_path):
        img = cv2.imread(filename)
        resized = cv2.resize(img,(h, w))
        imgs.append(resized)
    return np.array(imgs)

def getLabels(labels_path):
    labels = []
    for filename in glob.glob(labels_path):
        with open(filename) as file:
            label = json.load(file)
            #labels.append([label['shape_type'], label['character']])
            labels.append([shape_label.index(label['shape_type'])])
            l.append([label['shape_type'], label['character']])
    return np.array(labels)

imgs_path = './shapes/*.jpg';
labels_path = './shapes/*.json';

data = getImgs(imgs_path, 32, 32)
label = keras.utils.to_categorical(getLabels(labels_path), num_classes=13)

model = create_model()
model.load_weights("./model_0.85175_0.87875.h5")

print("=============================")

pred_y = model.predict(data)
pred_y = pred_y.reshape(len(pred_y), num_objects, -1)
pred_shapes = np.argmax(pred_y[..., 4:4+num_shapes], axis=-1).astype(int)  # take max from probabilities

print("++++++++++++++++++++++++++++++++++")
for i in range(0, len(pred_shapes)):
    print("=================================")
    print(shape_labels[pred_shapes[i][0]])
    print(shape_labels[pred_shapes[i][1]])
    print("label: ")
    print(l[i])

