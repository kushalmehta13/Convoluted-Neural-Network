from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import h5py
import scipy
import numpy as np
import glob
import os

from tflearn.data_utils import build_hdf5_image_dataset

size1 = 128
size2 = 128

test_file = "predict.txt"
build_hdf5_image_dataset(test_file,image_shape=(size1,size2),mode='file',output_path='predict_dataset.h5',categorical_labels = True)
h5f = h5py.File("predict_dataset.h5","r")
X = h5f['X']


img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
# Convolutional network building
network = input_data(shape=[None, size1, size2, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
network = conv_2d(network, 10, 16, activation='relu')
network = conv_2d(network, 10, 16, activation='relu')
network = conv_2d(network, 10, 16, activation='relu')

network = max_pool_2d(network, 2)
network = max_pool_2d(network, 2)
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network,0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)
model = tflearn.DNN(network, tensorboard_verbose=0,best_checkpoint_path='.')
model.load('model.tflearn')

prediction = model.predict(X[()])
count = 0
for i in prediction:
    count += 1
    print("%d\n" %count)
    print('Probability of image having garbage:%f\n' %(i[0]))
    print('Probability of image NOT having garbage:%f\n' %(i[1]))
