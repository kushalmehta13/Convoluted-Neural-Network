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

# Width
size1 = 128
# Height
size2 = 128

# Contains path/to/image <label id>
# We are using the same text file named validation as this corresponds to the set of all images unseen by the CNN
# In a sense execution of this program is kind of redundant since the trainer already gives us metrics related to this
test_file = "validation.txt"
# Creates a hdf5 image dataset by referring to validation.txt
build_hdf5_image_dataset(test_file,image_shape=(size1,size2),mode='file',output_path='predict_dataset.h5',categorical_labels = True)
# Open the dataset in read mode
h5f = h5py.File("predict_dataset.h5","r")   
# Get the images in h5py format. It is of the form [number of images,width,height,channels]
X = h5f['X']


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
# Number of filters = 10, size of each filter = 16
network = conv_2d(network, 10, 16, activation='relu') 
network = conv_2d(network, 10, 16, activation='relu')
network = conv_2d(network, 10, 16, activation='relu')

# Number of units in max pool layer is 2
network = max_pool_2d(network, 2)
network = max_pool_2d(network, 2)
network = max_pool_2d(network, 2)

# Number of units used in this fully connected network is 512
network = fully_connected(network, 512, activation='relu')
# Avoids overfitting
network = dropout(network,0.5)
# Number of units used in this fully connected netword is 2
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.0001)
model = tflearn.DNN(network, tensorboard_verbose=0,best_checkpoint_path='.') 
# Loads the saved model
model.load('model.tflearn')

# Takes every image one by one and predicts the probability of the image belonging to every class
prediction = model.predict(X[()])
count_positive_true = 0
count_negative_true = 0

for i in range(len(prediction)):
	if (prediction[i][0] > prediction[i][1]):
		# Count it as positive only if it is truly positive
		# The number 50 represents the number of positive images given by us for testing the accuracy
		if (i < 50):
			count_positive_true += 1
	else:
		# Count it as positive only if it is truly negative
		# Another 50 negative images were also given
		if (i >= 50):
			count_negative_true += 1
			
# Print the accuracy of the CNN using the confusion matrix
printf("Accuracy of the trained CNN is:" + str((count_positive_true+count_negative_true)/100))
