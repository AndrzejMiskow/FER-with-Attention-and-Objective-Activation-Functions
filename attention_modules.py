import math

from tensorflow.keras import layers
import tensorflow as tf
from keras import backend


def squeeze_excitation_block(input_layer, out_dim, ratio=32, name=None):
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


def ECA_Net_block(input_layer, kernel_size=3, adaptive=False, name=None):
	"""ECA-Net: Efficient Channel Attention Block
    Args:
      input_layer: input tensor
      kernel_size: integer, default: 3, size of the kernel for the convolution
      name: string, block label
    Returns:
      Output A tensor for the ECA-Net attention block
    """
	if adaptive:
		b = 1
		gamma = 2
		channels = input_layer.shape[-1]
		kernel_size = int(abs((math.log2(channels) + b / gamma)))
		if (kernel_size % 2) == 0:
			kernel_size = kernel_size + 1
		else:
			kernel_size = kernel_size

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

