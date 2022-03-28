from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Activation


def residual_block_v1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None, grayscale=True):
	"""A residual block for ResNetV1
    Args:
      x: input tensor.
      filters: array, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True, otherwise identity shortcut.
      name: string, block label.
      grayscale: bool, states when the input tensor is RGB or Grayscale.
    Returns:
      Output tensor for the residual block.
    """
	batch_axis = 1
	filters1, filters2, filters3 = filters

	if conv_shortcut:
		shortcut = layers.Conv2D(
			filters3, 1, strides=stride, name=name + '_Shortcut_Conv')(x)
		shortcut = layers.BatchNormalization(
			axis=batch_axis, epsilon=1.001e-5, name=name + '_Shortcut_BN')(shortcut)
	else:
		shortcut = x

	x = layers.Conv2D(filters1, 1, strides=stride, name=name + '_1_Conv')(x)
	x = layers.BatchNormalization(
		axis=batch_axis, epsilon=1.001e-5, name=name + '_1_BN')(x)
	x = layers.Activation('relu', name=name + '_1_Relu')(x)

	x = layers.Conv2D(
		filters2, kernel_size, padding='SAME', name=name + '_2_Conv')(x)
	x = layers.BatchNormalization(
		axis=batch_axis, epsilon=1.001e-5, name=name + '_2_BN')(x)
	x = layers.Activation('relu', name=name + '_2_Relu')(x)

	x = layers.Conv2D(filters3, 1, name=name + '_3_Conv')(x)
	x = layers.BatchNormalization(
		axis=batch_axis, epsilon=1.001e-5, name=name + '_3_BN')(x)

	x = layers.Add(name=name + '_Add')([shortcut, x])
	x = layers.Activation('relu', name=name + '_Output')(x)

	return x


def group_residuals_v1(x, filters, blocks, stride1=2, name=None):
	"""A group of stacked residual blocks for ResNetV1
    Args:
      x: tensor, input
      filters: array, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, group of blocks label.
    Returns:
      Output tensor for the stacked blocks.
    """

	x = residual_block_v1(x, filters, stride=stride1, name=name + '_Block1')

	for i in range(2, blocks + 1):
		x = residual_block_v1(x, filters, conv_shortcut=False, name=name + '_Block' + str(i))
	return x


def residual_block_v2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
	"""A residual block for ResNetV2
    Args:
      x: input tensor.
      filters: array, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True, otherwise identity shortcut.
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """

	batch_axis = 1

	filters1, filters2, filters3 = filters

	# ResNetV2 uses pre-activation which makes it perform better on deeper networks
	pre_activation = layers.BatchNormalization(
		axis=batch_axis, epsilon=1.001e-5, name=name + '_Pre-Activation_BN')(x)
	pre_activation = layers.Activation('relu', name=name + '_Pre-Activation_Relu')(pre_activation)

	if conv_shortcut:
		shortcut = layers.Conv2D(
			filters3, 1, strides=stride, name=name + '_Shortcut_Conv')(pre_activation)
	else:
		shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

	x = layers.Conv2D(filters1, 1, strides=1, use_bias=False,
					  name=name + '_1_Conv')(pre_activation)
	x = layers.BatchNormalization(
		axis=batch_axis, epsilon=1.001e-5, name=name + '_1_BN')(x)
	x = layers.Activation('relu', name=name + '_1_Relu')(x)

	x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_Pad')(x)
	x = layers.Conv2D(filters2, kernel_size, strides=stride,
					  use_bias=False, name=name + '_2_Conv')(x)
	x = layers.BatchNormalization(
		axis=batch_axis, epsilon=1.001e-5, name=name + '_2_BN')(x)
	x = layers.Activation('relu', name=name + '_2_Relu')(x)

	x = layers.Conv2D(filters3, 1, name=name + '_3_Conv')(x)
	x = layers.Add(name=name + '_Add_Output')([shortcut, x])

	return x


def group_residuals_v2(x, filters, blocks, stride1=2, name=None):
	"""A group of stacked residual blocks for ResNetV2
    Args:
      x: tensor, input
      filters: array, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, group of blocks label.
    Returns:
      Output tensor for the stacked blocks.
    """

	x = residual_block_v2(x, filters, conv_shortcut=True, name=name + '_Block1')
	for i in range(2, blocks):
		x = residual_block_v2(x, filters, name=name + '_Block' + str(i))
	# stride is changed here due to the pre-activation in the next layer
	x = residual_block_v2(x, filters, stride=stride1, name=name + '_Block' + str(blocks))
	return x
