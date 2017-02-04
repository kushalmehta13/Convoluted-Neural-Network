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


size1 = 128 #width
size2 = 128 #height


test_file = "predict.txt"   #Contains path/to/image <label id>
build_hdf5_image_dataset(test_file,image_shape=(size1,size2),mode='file',output_path='predict_dataset.h5',categorical_labels = True) #creates a hdf5 image dataset by referring to predict.txt
h5f = h5py.File("predict_dataset.h5","r")   #open the dataset in read mode
X = h5f['X']    # Get the images in h5py format. It is of the form [number of images,width,height,channels]


#Prepocess images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
network = input_data(shape=[None, size1, size2, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
network = conv_2d(network, 10, 16, activation='relu') #number of filters = 10, size of each filter = 16
network = conv_2d(network, 10, 16, activation='relu')
network = conv_2d(network, 10, 16, activation='relu')

network = max_pool_2d(network, 2)   #Number of units in max pool layer is 2
network = max_pool_2d(network, 2)
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')  #Number of units used in this fully connected network is 512
network = dropout(network,0.5)  #Avoids overfitting
network = fully_connected(network, 2, activation='softmax') #Number of units used in this fully connected netword is 2
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.0001)
model = tflearn.DNN(network, tensorboard_verbose=0,best_checkpoint_path='.') 
model.load('model.tflearn') #Loads the saved model

prediction = model.predict(X[()]) #takes every image one by one and predicts the probability of the image belonging to every class
count = 0
for i in prediction:
    count += 1
    print("%d\n" %count)
    print('Probability of image having garbage:%f\n' %(i[0]))
    print('Probability of image NOT having garbage:%f\n' %(i[1]))
