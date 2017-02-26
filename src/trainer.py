from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf
#import tflearn.helpers.summarizer as s
import h5py
import scipy
import numpy as np

from tflearn.data_utils import build_hdf5_image_dataset

# tflearn.summaries.get_summary('image','images')
# This file has been explained in Instructions.txt
dataset_file = 'training.txt'
test_dataset_file = 'validation.txt'
size1 = 128
size2 = 128

# Building the h5py dataset
build_hdf5_image_dataset(dataset_file,image_shape=(size1,size2),mode='file',output_path='training_dataset.h5',categorical_labels = True)
h5f = h5py.File('training_dataset.h5','r')

build_hdf5_image_dataset(test_dataset_file,image_shape=(size1,size2),mode='file',output_path='validation_dataset.h5',categorical_labels = True)
h5f1 = h5py.File('validation_dataset.h5','r')

# Reading from the dataset
# 'X' are all the images and 'Y' are all the labels corresponding to those images
X = h5f['X']
Y = h5f['Y']
X_test = h5f1['X']
Y_test = h5f1['Y']

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation - flipping and randomly rotating images to create more data
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Add the summary file writer
writer = tf.summary.FileWriter('/tmp/tflearn_logs/garbage_test/',graph=None,max_queue=10,flush_secs=120,graph_def=None)

# Convolutional network building
network = input_data(shape=[None, size1, size2, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
network = conv_2d(network, 20, 5, activation='relu')
summary_output = tf.summary.image("Convolution 1", network, max_outputs=4)
network=  max_pool_2d(network, 2)
network = conv_2d(network, 20, 7, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 20, 9, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 20, 11,activation='relu')
network = max_pool_2d(network, 2)

#network = max_pool_2d(network, 2)
#network = max_pool_2d(network, 2)
#network = max_pool_2d(network, 2)

network = fully_connected(network, 512, activation='relu')
# Dropout to take care of overfitting
network = dropout(network, 0.5)

network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer = 'adam',loss = 'categorical_crossentropy',learning_rate =0.0002)

#writer.add_summary(summary_output)
# Train using classifier
# Gonna remove tensorboard_verbose parameter
model = tflearn.DNN(network, best_checkpoint_path='.')
model.fit(X, Y, n_epoch=10, shuffle=True, validation_set = (X_test,Y_test), show_metric=True, batch_size=96, run_id='garbage_test')

#Save the model
model.save("model.tflearn")
