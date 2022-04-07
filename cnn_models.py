import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adamax
from keras import backend

from layers import *
from attention_modules import *


def gpu_check():
	print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def data_augmentation(input_img):
	input_img = layers.RandomFlip("horizontal")(input_img)
	input_img = layers.RandomRotation(0.2)(input_img)
	input_img = layers.RandomZoom(0.2)(input_img)
	# input_img = layers.RandomHeight(0.2)(input_img)
	# input_img = layers.RandomWidth(0.2)(input_img)
	input_img = layers.RandomContrast(0.3)(input_img)
	# value_range is between 0.0 and 1.0 if image  has been rescaled otherwise 0 to 255
	# input_img = layers.RandomBrightness(factor=0.3, value_range=(0.0, 1.0))(input_img)
	return input_img


def model_vgg16(img_height=48,
				img_width=48,
				a_hidden='relu',  # Hidden activation
				a_output='softmax',  # Output activation
				attention="",
				num_classes=7
				):
	"""Function to output the VGG16 CNN Model
       Args:
          img_height: integer,default '48', input image height
          img_width: integer,default '48', input image width
          a_hidden: string,default 'relu', activation function used for hidden layerss
          a_output: string, default 'softmax', output activation function
          num_classes: integer, default 7,states the number of classes
        Returns:
          Output A `keras.Model` instance.
    """
	# Input
	input_img = Input(shape=(img_height, img_width, 1), name="img")

	# 1st Conv Block
	x = Conv2D(filters=64, kernel_size=3, padding='same', activation=a_hidden, name="Conv1.1")(input_img)
	x = Conv2D(filters=64, kernel_size=3, padding='same', activation=a_hidden, name="Conv1.2")(x)
	if attention == "":
		x = x
	else:
		x = select_attention(x, 64, block_name=attention, layer_name="Conv1")
	x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_1")(x)

	# 2nd Conv Block
	x = Conv2D(filters=128, kernel_size=3, padding='same', activation=a_hidden, name="Conv2.1")(x)
	x = Conv2D(filters=128, kernel_size=3, padding='same', activation=a_hidden, name="Conv2.2")(x)
	if attention == "":
		x = x
	else:
		x = select_attention(x, 128, block_name=attention, layer_name="Conv2")
	x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_2")(x)

	# 3rd Conv block
	x = Conv2D(filters=256, kernel_size=3, padding='same', activation=a_hidden, name="Conv3.1")(x)
	x = Conv2D(filters=256, kernel_size=3, padding='same', activation=a_hidden, name="Conv3.2")(x)
	x = Conv2D(filters=256, kernel_size=3, padding='same', activation=a_hidden, name="Conv3.3")(x)
	if attention == "":
		x = x
	else:
		x = select_attention(x, 256, block_name=attention, layer_name="Conv3")
	x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_3")(x)

	# 4th Conv block
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv4.1")(x)
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv4.2")(x)
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv4.3")(x)
	if attention == "":
		x = x
	else:
		x = select_attention(x, 512, block_name=attention, layer_name="Conv4")
	x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_4")(x)

	# 5th Conv block
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv5.1")(x)
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv5.2")(x)
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv5.3")(x)
	if attention == "":
		x = x
	else:
		x = select_attention(x, 512, block_name=attention, layer_name="Conv5")
	x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_5")(x)

	# Fully connected layers
	x = Flatten()(x)
	x = Dense(units=4096, activation=a_hidden, name="Dense1")(x)
	x = Dense(units=4096, activation=a_hidden, name="Dense2")(x)

	output = Dense(units=num_classes, activation=a_output, name="DenseFinal")(x)
	if attention == "":
		model_name = "VGG16"
	else:
		model_name = "VGG16" + "_" + attention

	model = Model(inputs=input_img, outputs=output, name=model_name)

	return model


def model_vgg19(img_height=48,
				img_width=48,
				a_hidden='relu',  # Hidden activation
				a_output='softmax',  # Output activation
				attention="",
				num_classes=7):
	"""Function to output the VGG19 CNN Model
       Args:
          img_height: integer,default '48', input image height
          img_width: integer,default '48', input image width
          a_hidden: string,default 'relu', activation function used for hidden layerss
          a_output: string, default 'softmax', output activation function
          num_classes: integer, default 7,states the number of classes
        Returns:
          Output A `keras.Model` instance.
    """
	# Input
	input_img = Input(shape=(img_height, img_width, 1), name="img")

	# 1st Conv Block
	x = Conv2D(filters=64, kernel_size=3, padding='same', activation=a_hidden, name="Conv1.1")(input_img)
	x = Conv2D(filters=64, kernel_size=3, padding='same', activation=a_hidden, name="Conv1.2")(x)
	if attention == "":
		x = x
	else:
		x = select_attention(x, 64, block_name=attention, layer_name="Conv1")
	x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_1")(x)

	# 2nd Conv Block
	x = Conv2D(filters=128, kernel_size=3, padding='same', activation=a_hidden, name="Conv2.1")(x)
	x = Conv2D(filters=128, kernel_size=3, padding='same', activation=a_hidden, name="Conv2.2")(x)
	if attention == "":
		x = x
	else:
		x = select_attention(x, 128, block_name=attention, layer_name="Conv2")
	x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_2")(x)

	# 3rd Conv block
	x = Conv2D(filters=256, kernel_size=3, padding='same', activation=a_hidden, name="Conv3.1")(x)
	x = Conv2D(filters=256, kernel_size=3, padding='same', activation=a_hidden, name="Conv3.2")(x)
	x = Conv2D(filters=256, kernel_size=3, padding='same', activation=a_hidden, name="Conv3.3")(x)
	x = Conv2D(filters=256, kernel_size=3, padding='same', activation=a_hidden, name="Conv3.4")(x)
	if attention == "":
		x = x
	else:
		x = select_attention(x, 256, block_name=attention, layer_name="Conv3")
	x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_3")(x)

	# 4th Conv block
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv4.1")(x)
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv4.2")(x)
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv4.3")(x)
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv4.4")(x)
	if attention == "":
		x = x
	else:
		x = select_attention(x, 512, block_name=attention, layer_name="Conv4")
	x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_4")(x)

	# 5th Conv block
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv5.1")(x)
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv5.2")(x)
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv5.3")(x)
	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=a_hidden, name="Conv5.4")(x)
	if attention == "":
		x = x
	else:
		x = select_attention(x, 512, block_name=attention, layer_name="Conv5")
	x = MaxPool2D(pool_size=2, strides=2, padding='same', name="MaxPool2D_5")(x)

	# Fully connected layers
	x = Flatten()(x)
	x = Dense(units=4096, activation=a_hidden, name="Dense1")(x)
	x = Dense(units=4096, activation=a_hidden, name="Dense2")(x)

	output = Dense(units=num_classes, activation=a_output, name="DenseFinal")(x)

	if attention == "":
		model_name = "VGG19"
	else:
		model_name = "VGG19" + "_" + attention

	model = Model(inputs=input_img, outputs=output, name=model_name)

	return model


def model_ResNet_V1(
		model="ResNet50",
		img_height=48,
		img_width=48,
		a_output='softmax',
		pooling='avg',
		attention="",
		num_classes=7):
	"""Function that is able to return different ResNet V1 Models
       Args:
       	  model: string, default 'ResNet50', select which ResNet model to use ResNet50 , ResNet101 or ResNet152
          img_height: integer,default '48', input image height
          img_width: integer,default '48', input image width
          a_output: string, default 'softmax', output activation function
          pooling: string,default 'avg', pooling used for the final layer either 'avg' or 'max'
          attention: string, default '', select which Attention block to use SEnet , ECANet or CBAM
          num_classes: integer, default 7,states the number of classes
        Returns:
          Output A `keras.Model` instance.
    """
	# Input
	batch_axis = 1

	input_img = Input(shape=(img_height, img_width, 1), name="img")

	if model == "ResNet50":
		num_blocks = [3, 4, 6, 3]
	elif model == "ResNet18":
		num_blocks = [2, 2, 2, 2]
	elif model == "ResNet101":
		num_blocks = [3, 4, 23, 3]
	elif model == "ResNet152":
		num_blocks = [3, 8, 36, 3]

	# Data Augmentation
	input_img = data_augmentation(input_img)

	# Conv_1
	x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='Conv1_Pad')(input_img)
	x = layers.Conv2D(64, 7, strides=2, name='Conv1')(x)
	x = layers.BatchNormalization(axis=batch_axis, epsilon=1.001e-5, name='Conv1_BN')(x)
	x = layers.Activation('relu', name='Conv1_relu')(x)

	x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='MaxPool2D_1_Pad')(x)
	x = layers.MaxPooling2D(3, strides=2, name='MaxPool2D_1')(x)

	# Residual Stacked Blocks
	x = group_residuals_v1(x, [64, 64, 256], num_blocks[0], stride1=1, name='Conv2', attention=attention)
	x = group_residuals_v1(x, [128, 128, 512], num_blocks[1], name='Conv3', attention=attention)
	x = group_residuals_v1(x, [256, 256, 1024], num_blocks[2], name='Conv4', attention=attention)
	x = group_residuals_v1(x, [512, 512, 2048], num_blocks[3], name='Conv5', attention=attention)

	# Output
	if pooling == 'avg':
		x = layers.GlobalAveragePooling2D(name='AvgPool2D_Final')(x)
	else:
		x = layers.GlobalMaxPooling2D(name='MaxPool2D_Final')(x)

	output = layers.Dense(num_classes, activation=a_output, name='DenseFinal')(x)

	if attention == "":
		model = model
	else:
		model = model + "_" + attention

	model = Model(inputs=input_img, outputs=output, name="ResNet_V1" + model)
	return model


def model_ResNet_V2(
		model="ResNet50",
		img_height=48,
		img_width=48,
		a_output='softmax',
		pooling='avg',
		attention="",
		num_classes=7):
	"""Function that is able to return different Resnet V2 models
       Args:
       	  model: string, default 'ResNet50', select which ResNet model to use ResNet50 , ResNet101 or ResNet152
          img_height: integer,default '48', input image height
          img_width: integer,default '48', input image width
          a_output: string, default 'softmax', output activation function
          pooling: string,default 'avg', pooling used for the final layer either 'avg' or 'max'
          attention: string, default '', select which Attention block to use SEnet , ECANet or CBAM
          num_classes: integer, default 7,states the number of classes
        Returns:
          Output A `keras.Model` instance.
    """
	# Input
	batch_axis = 1
	input_img = Input(shape=(img_height, img_width, 1), name="img")

	if model == "ResNet50":
		num_blocks = [3, 4, 6, 3]
	elif model == "ResNet101":
		num_blocks = [3, 4, 23, 3]
	elif model == "ResNet152":
		num_blocks = [3, 8, 36, 3]

	# input_img = data_augmentation(input_img)

	# Conv_1
	x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='Conv1_Pad')(input_img)
	x = layers.Conv2D(64, 7, strides=2, use_bias=True, name='Conv1')(x)

	x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='MaxPool2D_1_Pad')(x)
	x = layers.MaxPooling2D(3, strides=2, name='MaxPool2D_1')(x)

	# Residual Stacked Blocks
	x = group_residuals_v2(x, [64, 64, 256], num_blocks[0], stride1=2, name='Conv2', attention=attention)
	x = group_residuals_v2(x, [128, 128, 512], num_blocks[1], stride1=2, name='Conv3', attention=attention)
	x = group_residuals_v2(x, [256, 256, 1024], num_blocks[2], stride1=2, name='Conv4', attention=attention)
	x = group_residuals_v2(x, [512, 512, 2048], num_blocks[3], stride1=1, name='Conv5', attention=attention)

	# Due to pre-activation we need to apply activation to last conv block
	x = layers.BatchNormalization(axis=batch_axis, epsilon=1.001e-5, name='Final_BN')(x)
	x = layers.Activation('relu', name='Final_Relu')(x)

	# Output
	if pooling == 'avg':
		x = layers.GlobalAveragePooling2D(name='AvgPool2D_Final')(x)
	else:
		x = layers.GlobalMaxPooling2D(name='MaxPool2D_Final')(x)

	output = layers.Dense(num_classes, activation=a_output, name='DenseFinal')(x)

	if attention == "":
		model = model
	else:
		model = model + "_" + attention

	model = Model(inputs=input_img, outputs=output, name="ResNet50_V2" + model)
	return model


def test_attention(
		img_height=48,
		img_width=48,
		a_output='softmax',
		pooling='avg',
		grayscale=True,
		num_classes=7):
	if grayscale:
		input_img = Input(shape=(img_height, img_width, 1), name="img")
	else:
		input_img = Input(shape=(img_height, img_width, 3), name="img")
		input_img = layers.experimental.preprocessing.Rescaling(1. / 255)(input_img)

	x = BAM_block(input_img, filter_num=48 * 3, reduction_ratio=4, dilution_conv=2, name="BAM")

	# Output
	if pooling == 'avg':
		x = layers.GlobalAveragePooling2D(name='AvgPool2D_Final')(x)
	else:
		x = layers.GlobalMaxPooling2D(name='MaxPool2D_Final')(x)

	output = layers.Dense(num_classes, activation=a_output, name='DenseFinal')(x)

	model = Model(inputs=input_img, outputs=output, name="BAM")

	return model
