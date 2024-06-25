from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Dropout, Dropout
from keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adam
from PIL import Image
from keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def gen_labels():
    labels = {0 : "Bakkat Makki", 1 : "Chi Jieurigi", 2 : "Dwi Chagi", 3 : "Eolgeol Makki", 4 : "Up Chagi", 5 : "Yeop Jireugi"}
    return labels

def preprocess(image):
    image = np.array(image.resize((300, 300), Image.ANTIALIAS))
    image = np.array(image, dtype='uint8')
    image = np.array(image)/255.0
    return image


def CNN_Model():
    input_shape = (300, 300, 3)
    num_classes=6
    model = Sequential()    
    model.add(Conv2D(32, kernel_size=(3,3), padding='same', strides=(1, 1), input_shape = input_shape, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=2))

    model.add(Conv2D(32, kernel_size = (3,3), padding='same', strides=(1, 1),activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=2))

    model.add(Conv2D(32, kernel_size = (3,3), padding='same', strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    return model

