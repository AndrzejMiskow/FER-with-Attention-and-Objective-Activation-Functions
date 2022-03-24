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
    if grayscale:
        batch_axis = 1
    else:
        batch_axis = 3

    filters1, filters2, filters3 = filters

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters3, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=batch_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters1, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=batch_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters2, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=batch_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(filters3, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
        axis=batch_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)

    return x


def group_residuals_v1(x, filters, blocks, stride1=2, name=None, grayscale=True):
    """A group of stacked residual blocks for ResNetV1
    Args:
      x: tensor, input
      filters: array, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, group of blocks label.
      grayscale: bool, states when the input tensor is RGB or Grayscale
    Returns:
      Output tensor for the stacked blocks.
    """

    x = residual_block_v1(x, filters, stride=stride1, name=name + '_Block1', grayscale=grayscale)

    for i in range(2, blocks + 1):
        x = residual_block_v1(x, filters, conv_shortcut=False, name=name + '_Block' + str(i), grayscale=grayscale)

    return x

