import glob
import os
import re

def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
	return [ atoi(c) for c in re.split('(\d+)', text) ]

# We have implemented 2 classes, hence we have 4 folders 2 classes in Testing and Training each
# Your directory structure may vary and this will vary accordingly
directories = ["../Dataset/Testing/Positive/*","../Dataset/Testing/Negative/*","../Dataset/Training/Positive/*","../Dataset/Training/Negative/*"]
# We are generating two .txt files namely training and validation 
# We are using vallidation for checking the validation accuracy while training the CNN
# Alternatively a percentage of the training set can also be set as the validation set
# We have two labels 0 - positive and 1 - Negative
generated = ""
label = 0

for i in range(len(directories)):
	if (i == 0 or i == 2):
		label = 0
	else:
		label = 1

	if (i == 0 or i == 1):
		generated = "training.txt"
	else:
		generated = "validation.txt"

	l = []
	f = open(generated,'a')
	for filename in glob.glob(directories[i]):
		l.append(filename);
		
	l.sort(key=natural_keys)
	# The output of each of those text files is of the form <path_to_image> <label_of_that_image> - Yes, there is a space in between
	for x in l:
		f.write(x+' '+str(label)+'\n')
	f.close();
