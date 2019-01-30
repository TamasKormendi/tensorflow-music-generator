# Code adapted from https://github.com/chrisdonahue/wavegan/blob/master/wavegan.py

import tensorflow as tf
import math

def num_filters(block_id, fmap_base=8192, fmap_decay=1.0, fmap_max=512):
    return int(min(fmap_base / math.pow(2.0, block_id * fmap_decay), fmap_max))

def block_name(block_id):
    return "progressive_block_{}".format(block_id)

# Now TF also has https://www.tensorflow.org/api_docs/python/tf/contrib/nn/conv1d_transpose which might be worth a look
def conv1d_transpose(
        inputs,
        filters,
        kernel_width,
        stride=4,
        padding="same",
        upsample="zeros"):

    if upsample == "zeros":
        return tf.layers.conv2d_transpose(
            tf.expand_dims(inputs, axis=1),
            filters,
            (1, kernel_width),
            strides=(1, stride),
            padding=padding
        )[:, 0]
    elif upsample == "nn":  # nn stands for nearest neighbour
        batch_size = tf.shape(inputs)[0]
        _, width, number_of_channels = inputs.get_shape().as_list()

        upsampled_inputs = inputs

        upsampled_inputs = tf.expand_dims(upsampled_inputs, axis=1)
        upsampled_inputs = tf.image.resize_nearest_neighbor(upsampled_inputs, [1, width * stride])
        upsampled_inputs = upsampled_inputs[:, 0]

        return tf.layers.conv1d(
            upsampled_inputs,
            filters,
            kernel_width,
            1,
            padding=padding)
    else:
        raise NotImplementedError


"""
    Input: 100 random values with shape [None, 100] - called "z" in original code
    Output: 16384 sound samples with shape [None, 16384, 1]
    The "None"s are the batch size, 1 in the output is the number of channels
"""

def GANGenerator(
        input,
        kernel_len=25,
        dim=64,          # Simply a multiplier for the number of feature maps
        use_batchnorm=False,
        upsample="zeros",
        train=False,
        num_blocks=None,
        channels=1
        ):

    def to_output(x):
        return tf.layers.conv1d(
            x,
            channels,
            kernel_len,
            padding="SAME",
            activation=tf.nn.tanh
        )

    batch_size = tf.shape(input)[0]

    if use_batchnorm:
        batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
    else:
        batchnorm = lambda x: x

    # Reshape layer/projection
    output = input

    # For now, try with 512 fmaps initially

    # [100] to [16, 512]
    # 16-length, 512 channels
    with tf.variable_scope("input_project"):
        output = tf.layers.dense(output, 4 * 4 * dim * 8)
        output = tf.reshape(output, [batch_size, 16, dim * 8])
        output = batchnorm(output)

    output = tf.nn.relu(output)

    # Every block quadruples the amount of samples

    # Unlike in the PGGAN repo, the whole network is built in the loop
    # Note: for now it does not do any blending
    # TODO: figure out how to blend audio between training stages
    for block_id in range(1, num_blocks + 1):
        with tf.variable_scope(block_name(block_id)):
            output = conv1d_transpose(output, num_filters(block_id), kernel_len, 4, upsample=upsample)
            output = batchnorm(output)
        output = tf.nn.relu(output)

    # Scoped for the last block so it does not get used when a bigger network runs
    with tf.variable_scope(block_name(num_blocks)):
        output = to_output(output)

    # Need to use an identity matrix because otherwise the batchnorm moving values don't get
    # placed in the update ops and don't get computed
    if train and use_batchnorm:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if len(update_ops) != 10:
            raise Exception("Other update ops found in graph")
        with tf.control_dependencies(update_ops):
            output = tf.identity(output)

    return output

# Leaky ReLU
def lrelu(inputs, alpha=0.2):
    return tf.maximum(alpha * inputs, inputs)

# The radius is the uniform distribution [-radius, radius] by which the activation
# phases are perturbed for each layer
def apply_phaseshuffle(input, radius, pad_type="reflect"):
    batch_size, input_length, number_of_channels = input.get_shape().as_list()

    phase = tf.random_uniform([], minval=-radius, maxval=radius + 1, dtype=tf.int32)
    pad_left = tf.maximum(phase, 0)
    pad_right = tf.maximum(-phase, 0)

    phase_start = pad_right

    output = tf.pad(input, [[0, 0], [pad_left, pad_right], [0, 0]], mode=pad_type)

    output = output[:, phase_start:phase_start+input_length]
    output.set_shape([batch_size, input_length, number_of_channels])

    return output

"""
    Input: 16384 sound samples:
    [None, 16384, 1] - [batch_size, samples, channels]
    Output: linear value
"""

def GANDiscriminator(
        input,
        kernel_len=25,
        dim=64,
        use_batchnorm=False,
        phaseshuffle_rad=0,
        num_blocks=None):

    batch_size = tf.shape(input)[0]

    if use_batchnorm:
        batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
    else:
        batchnorm = lambda x: x

    if phaseshuffle_rad > 0:
        phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
    else:
        phaseshuffle = lambda x: x


    output = input

    # No blending yet
    # Whole network is constructed in the loop

    for block_id in range(num_blocks, 0, -1):
        with tf.variable_scope(block_name(block_id)):
            output = tf.layers.conv1d(output, num_filters(block_id), kernel_len, 4, padding="SAME")
        output = lrelu(output)
        output = phaseshuffle(output)

    # Final conv output should be [16, 512]
    output = tf.reshape(output, [batch_size, 4 * 4 * dim * 8])

    # Compute a single output
    with tf.variable_scope("output"):
        output = tf.layers.dense(output, 1)[:, 0]

    # No need to add the batchnorm ops to the training ops since
    # moving statistics are only used in inference mode
    # the discriminator is only for training

    return output