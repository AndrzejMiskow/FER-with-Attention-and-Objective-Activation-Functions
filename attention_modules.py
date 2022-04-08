import math

from tensorflow.keras import layers
import tensorflow as tf
from keras import backend


def squeeze_excitation_block(input_layer, out_dim, ratio=4, name=None):
	"""Squeeze and Extraction block
   Args:
      input_layer: input tensor
      out_dim: integer, output dimension for the model
      ratio: integer, reduction ratio for the number of neurons in the hidden layers
      name: string, block label
    Returns:
      Output A tensor for the squeeze and excitation block
    """

	#  Get the number of channels of the input characteristic graph
	in_channel = input_layer.shape[-1]

	#  Global average pooling [h,w,c]==>[None,c]
	squeeze = layers.GlobalAveragePooling2D(name=name + "_Squeeze_GlobalPooling")(input_layer)
	# [None,c]==>[1,1,c]
	squeeze = layers.Reshape(target_shape=(1, 1, in_channel))(squeeze)

	excitation = layers.Dense(units=out_dim / ratio, name=name + "_Excitation_FC_1")(squeeze)
	excitation = layers.Activation('relu', name=name + '_Excitation_FC_Relu_1')(excitation)

	excitation = layers.Dense(out_dim, name=name + "_Excitation_FC_2")(excitation)
	excitation = layers.Activation('sigmoid', name=name + '_Excitation_FC_Relu_2')(excitation)
	excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

	scale = layers.multiply([input_layer, excitation])

	return scale


def ECA_Net_block(input_layer, kernel_size=3, name=None):
	"""ECA-Net: Efficient Channel Attention Block
    Args:
      input_layer: input tensor
      kernel_size: integer, default: 3, size of the kernel for the convolution
      name: string, block label
    Returns:
      Output A tensor for the ECA-Net attention block
    """
	squeeze = layers.GlobalAveragePooling2D(name=name + "_Squeeze_GlobalPooling")(input_layer)
	squeeze = tf.expand_dims(squeeze, axis=1)
	excitation = layers.Conv1D(filters=1,
							   kernel_size=kernel_size,
							   padding='same',
							   use_bias=False,
							   name=name + "_Excitation_Conv_1D")(squeeze)

	excitation = tf.expand_dims(tf.transpose(excitation, [0, 2, 1]), 3)
	excitation = tf.math.sigmoid(excitation)

	output = layers.multiply([input_layer, excitation])

	return output


def CBAM_block(input_layer, filter_num, reduction_ratio=32, kernel_size=7, name=None):
	"""CBAM: Convolutional Block Attention Module Block
    Args:
      input_layer: input tensor
      filter_num: integer, number of neurons in the hidden layers
      reduction_ratio: integer, default: 32,reduction ratio for the number of neurons in the hidden layers
      name: string, block label
    Returns:
      Output A tensor for the CBAM attention block
    """
	axis = -1

	# CHANNEL ATTENTION
	avg_pool = layers.GlobalAveragePooling2D(name=name + "_Chanel_AveragePooling")(input_layer)
	max_pool = layers.GlobalMaxPool2D(name=name + "_Chanel_MaxPooling")(input_layer)

	# Shared MLP
	dense1 = layers.Dense(filter_num // reduction_ratio, activation='relu', name=name + "_Chanel_FC_1")
	dense2 = layers.Dense(filter_num, name=name + "_Chanel_FC_2")

	avg_out = dense2(dense1(avg_pool))
	max_out = dense2(dense1(max_pool))

	channel = layers.add([avg_out, max_out])
	channel = layers.Activation('sigmoid', name=name + "_Chanel_Sigmoid")(channel)
	channel = layers.Reshape((1, 1, filter_num), name=name + "_Chanel_Reshape")(channel)

	channel_output = layers.multiply([input_layer, channel])

	# SPATIAL ATTENTION
	avg_pool2 = tf.reduce_mean(input_layer, axis=axis, keepdims=True)
	max_pool2 = tf.reduce_max(input_layer, axis=axis, keepdims=True)

	spatial = layers.concatenate([avg_pool2, max_pool2], axis=axis)

	# K = 7 achieves the highest accuracy
	spatial = layers.Conv2D(1, kernel_size=kernel_size, padding='same', name=name + "_Spatial_Conv2D")(spatial)
	spatial_out = layers.Activation('sigmoid', name=name + "_Spatial_Sigmoid")(spatial)

	CBAM_out = layers.multiply([channel_output, spatial_out])

	return CBAM_out


def BAM_block(input_layer, filter_num, reduction_ratio, dilution_conv=2, name=None):
	axis = -1

	# CHANNEL ATTENTION
	# Global Average Pool to get channel vector [C x 1 x 1]
	# avg_pool = layers.GlobalAveragePooling2D(name=name + "_Chanel_AveragePooling")(input_layer)
	avg_pool = tf.reduce_mean(input_layer, axis=[1, 2], keepdims=True)

	# gate_chanels = [filter_num]
	# gate_chanels += [filter_num // reduction_ratio] * dilution_conv
	# gate_chanels += [filter_num]
	#
	# flaten = layers.Flatten()(avg_pool)
	# for i in range(len(gate_chanels) - 2):
	# 	fc = layers.Dense(gate_chanels[i + 1])(flaten)
	# 	fc = layers.BatchNormalization(axis=axis, epsilon=1.001e-5, name=name + '_Chanel_FC1_BN')(fc)
	# 	fc = layers.Activation('relu')(fc)
	# channel_output = layers.Dense(gate_chanels[-1])(fc)
	# channel_output = layers.Reshape((1, 1, filter_num), name=name + "_Chanel_Reshape")(channel_output)

	# Fully Connected Layer 1
	fc1 = layers.Dense(filter_num // reduction_ratio, name=name + "_Chanel_FC1")(avg_pool)
	fc1 = layers.BatchNormalization(axis=axis, epsilon=1.001e-5, name=name + '_Chanel_FC1_BN')(fc1)
	fc1 = layers.Activation('relu', name=name + "_Chanel_FC1_Relu")(fc1)

	# Fully Connected Layer 2
	fc2 = layers.Dense(filter_num, name=name + "_Chanel_FC2")(fc1)
	fc2 = layers.BatchNormalization(axis=axis, epsilon=1.001e-5, name=name + '_Chanel_FC2_BN')(fc2)
	channel_output = layers.Activation('relu', name=name + "_Chanel_FC2_Relu")(fc2)
	# channel_output = layers.Reshape((filter_num,1,1), name=name + "_Chanel_Reshape")(channel_output)


	# SPATIAL ATTENTION
	spatial = layers.Conv2D(filter_num // reduction_ratio, kernel_size=1, name=name + '_Spatial_1x1_Conv')(input_layer)
	spatial = layers.BatchNormalization(axis=axis, epsilon=1.001e-5, name=name + '_Spatial_BN1')(spatial)
	spatial = layers.Activation('relu', name=name + "_Spatial_1x1Conv_Relu")(spatial)

	for i in range(dilution_conv):
		spatial = layers.Conv2D(filter_num // reduction_ratio, kernel_size=3, padding="same",
								dilation_rate=dilution_conv,
								name=name + '_Spatial_3x3_Conv_' + str(i + 1))(spatial)
		spatial = layers.BatchNormalization(axis=axis, epsilon=1.001e-5,
											name=name + '_Spatial_3x3_Conv_BN_' + str(i + 1))(spatial)
		spatial = layers.Activation('relu', name=name + "_Spatial_3x3_Conv_Relu_" + str(i + 1))(spatial)

	spatial = layers.Conv2D(1, kernel_size=1, name=name + '_Spatial_Final_Conv')(spatial)
	spatial_output = layers.BatchNormalization(axis=axis, epsilon=1.001e-5, name=name + '_Spatial_Final_Conv_BN')(
		spatial)

	add = layers.multiply([channel_output, spatial_output])
	sigmoid = 1 + tf.math.sigmoid(add)
	output = sigmoid * input_layer

	return output
