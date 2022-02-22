import os
import pandas as pd

import detect_face

raw_data = "data/uncroped_data/"
processed_data = "data/processed_data/"
weights = "models/yolov5n-face.pt"


# Information on number of images in each folder
def number_of_images(path, folder_name):
	image = {}

	for file in os.listdir(path):
		category = path + file
		image[file] = len(os.listdir(category))  # Number of images in each category
	table = pd.DataFrame(image, index=[folder_name])

	return table


# function to crop faces out of images
def process_data(path):
	# Load pre-trained face recognition model
	detect_face.import_model(weights)

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


# clear_data_folders()
# number_of_images(raw_data, 'raw_data')
# clear_folder(raw_data)

# show the number of images in data folders
print(number_of_images(raw_data, "raw data"))
print(number_of_images(processed_data, "processed data"))

# process_data(raw_data)
