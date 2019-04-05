"""
The code in this file is based on WaveGAN v1: https://github.com/chrisdonahue/wavegan/tree/v1
and the Tensorflow Models implementation of PGGAN: https://github.com/tensorflow/models/tree/master/research/gan/progressive_gan

Code from both of these is adapted, modified and interwoven with new code so it would be impractical to point out which section of code is inspired by which.
Because of that this comment serves as the attribution.

If code is adapted from other sources it is explicitly pointed out.
"""

import tensorflow as tf
import math

# fmap means feature map
def num_filters(block_id, fmap_base=8192, fmap_decay=1.0, fmap_max=256, smooth_later_fmaps=True):

    # This comment block is valid for using the PGGAN fmap calculation method:
    # block_id + 1 is needed since this implementation does not exactly follow the
    # PGGAN implementation - first block outputs 64 samples, thus 8 blocks would be the maximum - 1024x1024
    # return int(min(fmap_base / math.pow(2.0, (block_id + 1) * fmap_decay), fmap_max))

    if smooth_later_fmaps:
        if block_id < 5:
            return 1024 // (2 ** block_id)
        else:
            base_filters = num_filters(4)

            working_block_id = block_id - 4
            # Ceiling division based on https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python/17511341#17511341
            exponent = -(-working_block_id // 2)
            divided_filters = base_filters // (2 ** (exponent - 1))

            if block_id % 2 != 0:
                return int(0.75 * divided_filters)
            else:
                return int(0.5 * divided_filters)
    else:
        return 1024 // (2 ** block_id)

def block_name(block_id):
    return "progressive_block_{}".format(block_id)

def sample_norm(samples, epsilon=1.0e-8):
    return samples * tf.rsqrt(tf.reduce_mean(tf.square(samples), axis=2, keepdims=True) + epsilon)

def conv1d_transpose(
        inputs,
        filters,
        kernel_width,
        stride=4,
        padding="same",
        upsample="zeros",
        trainable=True):

    if upsample == "zeros":
        return tf.layers.conv2d_transpose(
            tf.expand_dims(inputs, axis=1),
            filters,
            (1, kernel_width),
            strides=(1, stride),
            padding=padding,
            trainable=trainable
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
            padding=padding,
            trainable=trainable)
    else:
        raise NotImplementedError


"""
    Input: 100 random values with shape [None, 100] - called "z" in original code
    Output: Amount of samples corresponding to the output amount of num_blocks with shape [None, output_amount, channel_count]
    "None" corresponds to the batch_size
"""

def GANGenerator(
        input,
        kernel_len=25,
        dim=64,          # Simply a multiplier for the number of feature maps
        use_batchnorm=False,
        upsample="zeros",
        train=False,
        num_blocks=None,
        channels=1,
        freeze_early_layers=False,
        use_samplenorm=False
        ):

    # 1x1 output conv
    def to_output(x):
        return tf.layers.conv1d(
            x,
            channels,
            1,
            padding="SAME",
            activation=tf.nn.tanh
        )

    batch_size = tf.shape(input)[0]

    if use_batchnorm:
        batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
    else:
        batchnorm = lambda x: x

    if use_samplenorm:
        samplenorm = lambda x: sample_norm(x)
    else:
        samplenorm = lambda x: x

    output = input

    # Reshape layer/projection
    # [100] to [16, 1024]
    # 16-length, 1024 channels
    with tf.variable_scope("input_project"):
        output = tf.layers.dense(output, 4 * 4 * dim * 16)
        output = tf.reshape(output, [batch_size, 16, dim * 16])
        output = batchnorm(output)
    output = tf.nn.leaky_relu(output)
    output = samplenorm(output)

    # Every block quadruples the amount of samples

    # Unlike in the PGGAN repo, the whole network is built in the loop if the early layers are not frozen
    if not freeze_early_layers:
        for block_id in range(1, num_blocks + 1):
            with tf.variable_scope(block_name(block_id)):
                output = conv1d_transpose(output, num_filters(block_id), kernel_len, 4, upsample=upsample)
                output = batchnorm(output)
            output = tf.nn.leaky_relu(output)
            output = samplenorm(output)
    else:
        # Freeze layers from 1 until num_blocks - 1
        for block_id in range(1, num_blocks):
            with tf.variable_scope(block_name(block_id)):
                output = conv1d_transpose(output, num_filters(block_id), kernel_len, 4, upsample=upsample, trainable=False)
                output = batchnorm(output)
            output = tf.nn.leaky_relu(output)
            output = samplenorm(output)

        # Only make the last layer trainable
        with tf.variable_scope(block_name(num_blocks)):
            output = conv1d_transpose(output, num_filters(num_blocks), kernel_len, 4, upsample=upsample)
            output = batchnorm(output)
        output = tf.nn.leaky_relu(output)
        output = samplenorm(output)

    # Scoped for the last block so it does not get used when a bigger network runs
    with tf.variable_scope(block_name(num_blocks) + "_output"):
        output = to_output(output)

    # Need to use an identity matrix because otherwise the batchnorm moving values don't get
    # placed in the update ops and don't get computed
    # NOTE: DON'T USE BATCHNORM WITH WGAN-GP
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
    Input: Amount of samples corresponding to output of num_blocks.
    [None, sample_amount, 1] - [batch_size, samples, channels]
    Output: linear value
"""

def GANDiscriminator(
        input,
        kernel_len=25,
        dim=64,
        use_batchnorm=False,
        phaseshuffle_rad=0,
        num_blocks=None,
        freeze_early_layers=False):

    # Turns the 1/2 channel input into as many channels as the new top
    # layer would expect from the previous top layer
    def from_input(x, block_id):
        return tf.layers.conv1d(x, num_filters(block_id), 1, activation=tf.nn.leaky_relu)

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

    # Whole network is constructed in the loop if the layers are not frozen
    with tf.variable_scope(block_name(num_blocks) + "_input"):

        # Comment below and commented out code parts are valid for the PGGAN method of fmap calculation:
        # The +1 is needed so the kernel_shape is going to match what the new top layer expects
        # For example 64 input to 128 output channels, without the +1 it would be 128 in to 128 out
        # output = from_input(output, num_blocks + 1)

        output = from_input(output, num_blocks)

    if not freeze_early_layers:
        for block_id in range(num_blocks, 0, -1):
            with tf.variable_scope(block_name(block_id)):
                # output = tf.layers.conv1d(output, num_filters(block_id), kernel_len, 4, padding="SAME")
                output = tf.layers.conv1d(output, num_filters(block_id - 1), kernel_len, 4, padding="SAME")
            output = lrelu(output)
            if block_id > 1:
                output = phaseshuffle(output)
    else:
        # Construct trainable top block
        with tf.variable_scope(block_name(num_blocks)):
            # output = tf.layers.conv1d(output, num_filters(num_blocks), kernel_len, 4, padding="SAME")
            output = tf.layers.conv1d(output, num_filters(num_blocks - 1), kernel_len, 4, padding="SAME")
        output = lrelu(output)
        if num_blocks > 1:
            output = phaseshuffle(output)

        # Freeze bottom blocks - run the loop from num_blocks - 1
        for block_id in range(num_blocks - 1, 0, -1):
            with tf.variable_scope(block_name(block_id)):
                # output = tf.layers.conv1d(output, num_filters(block_id), kernel_len, 4, padding="SAME", trainable=False)
                output = tf.layers.conv1d(output, num_filters(block_id - 1), kernel_len, 4, padding="SAME", trainable=False)
            output = lrelu(output)
            if block_id > 1:
                output = phaseshuffle(output)

    # Final conv output should be [16, fmap_max]
    output = tf.reshape(output, [batch_size, -1])

    # Compute a single output
    with tf.variable_scope("output"):
        output = tf.layers.dense(output, 1)[:, 0]

    # No need to add the batchnorm ops to the training ops since
    # moving statistics are only used in inference mode
    # the discriminator is only for training

    return output
