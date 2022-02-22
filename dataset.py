import os
import pandas as pd
import splitfolders
from keras.preprocessing.image import ImageDataGenerator, load_img

import detect_face

raw_data = "data/uncroped_data/"
processed_data = "data/processed_data/"
weights = "models/yolov5n-face.pt"
train_data = "data/train"
test_data = "data/test"


# Information on number of images in each folder
def number_of_images(path, folder_name):
	image = {}

	for file in os.listdir(path):
		category = path + file
		image[file] = len(os.listdir(category))  # Number of images in each category
	table = pd.DataFrame(image, index=[folder_name])

	return table


# Clears all images from specified data folder
def clear_folder(path):
	for file in os.listdir(path):
		category = path + file
		for image in os.listdir(category):
			file_path = os.path.join(category, image)
			os.remove(file_path)


# function to clear all the data folders
def clear_data_folders():
	clear_folder(raw_data)
	clear_folder(processed_data)


# function to crop faces out of images
def process_data(path):
	# Load pre-trained face recognition model
	detect_face.import_model(weights)
	# Clear the folder
	clear_folder(processed_data)

	for file in os.listdir(path):
		category = path + file
		# Ignore hidden files
		if file.startswith('.'):
			continue
		else:
			for image in os.listdir(category):
				# For each image feed it into detect_face and place
				# it in processed_data categories folders
				input_path = os.path.join(category, image)
				output_path = processed_data + file + "/" + image
				detect_face.run_detect_face(input_path, 80, output_path)
				print(output_path)


def test_process_data(image_name):
	detect_face.import_model(weights)
	detect_face.run_detect_face(image_name, 80, "result.jpg")


# splits the processed data into training and test sets
def split_training_test(path):
	splitfolders.ratio(path, output="data/",
					   seed=1337, ratio=(.8, .2), group_prefix=None, move=False)

	# enables oversampling of imbalanced datasets, works only with --fixed
	# splitfolders.ratio(path, output="data/",
	# 				   seed=1337, ratio=(.8, .2), group_prefix=None, move=False, oversample=True)

	# Change folder name of val to test
	os.rename("data/val", "data/test")


def data_loader():
	train_datagen = ImageDataGenerator(horizontal_flip=True,
									   validation_split=0.2)

	training_set = train_datagen.flow_from_directory(train_data,
													 batch_size=64,
													 target_size=(48, 48),
													 shuffle=True,
													 color_mode='grayscale',
													 class_mode='categorical',
													 subset='training')

	validation_set = train_datagen.flow_from_directory(train_data,
													   batch_size=64,
													   target_size=(48, 48),
													   shuffle=True,
													   color_mode='grayscale',
													   class_mode='categorical',
													   subset='validation')

	test_datagen = ImageDataGenerator(horizontal_flip=True)

	test_set = test_datagen.flow_from_directory(test_data,
												batch_size=64,
												target_size=(48, 48),
												shuffle=True,
												color_mode='grayscale',
												class_mode='categorical')

	print(training_set.class_indices)

	return training_set, validation_set, test_set

# data_loader()

# modify training set to be split into training and validations set

#


# clear_data_folders()
# number_of_images(raw_data, 'raw_data')
# clear_folder(processed_data)

# show the number of images in data folders
# print(number_of_images(raw_data, "raw data"))
# print(number_of_images(processed_data, "processed data"))

# process_data(raw_data)

# test_process_data("test.jpeg")
