"""
This whole file is a slight rewrite of https://github.com/chrisdonahue/wavegan/blob/v1/wavegan.py and thus it is heavily based on it
so this comment serves as the attribution.
"""

import tensorflow as tf

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
        dim=64,          # Filter dimensionality TODO: look into this more and confirm it
        use_batchnorm=False,
        upsample="zeros",
        train=False):

    batch_size = tf.shape(input)[0]

    if use_batchnorm:
        batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
    else:
        batchnorm = lambda x: x

    # Reshape layer/projection
    output = input

    # [100] to [16, 1024]
    # 16-length, 1024 channels
    with tf.variable_scope("input_project"):
        output = tf.layers.dense(output, 4 * 4 * dim *16)
        output = tf.reshape(output, [batch_size, 16, dim * 16])
        output = batchnorm(output)

    output = tf.nn.relu(output)

    # Every conv layer quadruples the amount of samples
    # And halves the channels (aside from the last one)

    # Conv layer 0
    # [16, 1024] to [64, 512]
    with tf.variable_scope("upconv_0"):
        output = conv1d_transpose(output, dim * 8, kernel_len, 4, upsample=upsample)
        output = batchnorm(output)
    output = tf.nn.relu(output)

    # Conv layer 1
    # [64, 512] to [256, 256]
    with tf.variable_scope("upconv_1"):
        output = conv1d_transpose(output, dim * 4, kernel_len, 4, upsample=upsample)
        output = batchnorm(output)
    output = tf.nn.relu(output)

    # Conv layer 2
    # [256, 256] to [1024, 128]
    with tf.variable_scope("upconv_2"):
        output = conv1d_transpose(output, dim * 2, kernel_len, 4, upsample=upsample)
        output = batchnorm(output)
    output = tf.nn.relu(output)

    # Conv layer 3
    # [1024, 128] to [4096, 64]
    with tf.variable_scope("upconv_3"):
        output = conv1d_transpose(output, dim, kernel_len, 4, upsample=upsample)
        output = batchnorm(output)
    output = tf.nn.relu(output)

    # Conv layer 4
    # [4096, 64] to [16384, 1]
    with tf.variable_scope("upconv_4"):
        output = conv1d_transpose(output, 1, kernel_len, 4, upsample=upsample)
    output = tf.nn.tanh(output)

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
        phaseshuffle_rad=2):

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

    # Conv layer 0
    # [16384, 1] to [4096, 64]
    with tf.variable_scope("downconv_0"):
        output = tf.layers.conv1d(output, dim, kernel_len, 4, padding="SAME")
    output = lrelu(output)
    output = phaseshuffle(output)

    # Conv layer 1
    # [4096, 64] to [1024, 128]
    with tf.variable_scope("downconv_1"):
        output = tf.layers.conv1d(output, dim * 2, kernel_len, 4, padding="SAME")
        output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Conv layer 2
    # [1024, 128] to [256, 256]
    with tf.variable_scope("downconv_2"):
        output = tf.layers.conv1d(output, dim * 4, kernel_len, 4, padding="SAME")
        output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Conv layer 3
    # [256, 256] to [64, 512]
    with tf.variable_scope("downconv_3"):
        output = tf.layers.conv1d(output, dim * 8, kernel_len, 4, padding="SAME")
        output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Conv layer 4
    # [64, 512] to [16, 1024]
    with tf.variable_scope("downconv_4"):
        output = tf.layers.conv1d(output, dim * 16, kernel_len, 4, padding="SAME")
        output = batchnorm(output)
    output = lrelu(output)

    # Flatten the output
    output = tf.reshape(output, [batch_size, 4 * 4 * dim * 16])

    # Compute a single output
    with tf.variable_scope("output"):
        output = tf.layers.dense(output, 1)[:, 0]

    # No need to add the batchnorm ops to the training ops since
    # moving statistics are only used in inference mode
    # the discriminator is only for training

    return output
