import os
import time

import numpy as np
import tensorflow as tf

from model import GANGenerator, GANDiscriminator
import dataloader
SAMPLING_RATE = 16000
# 100 random inputs for the generator
G_INIT_INPUT_SIZE = 100
# TODO: PLACEHOLDER
D_UPDATES_PER_G_UPDATE = 5

# TODO: Will have to adjust this for prog growing
window_size = 16384
# TODO: PLACEHOLDER
batch_size = 64

# G = generator
# D = discriminator

def train(filepath):
    loader = dataloader.Dataloader(window_size, "audio/")

    # TODO: change this to a placeholder so can be used for looping through data
    x = loader.get_next()

    G_input = tf.random_uniform([batch_size, G_INIT_INPUT_SIZE], -1., 1., dtype=tf.float32)

    # Generator network
    with tf.variable_scope("G"):
        G_output = GANGenerator(G_input, train=True)
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G")

    # Discriminator with real input data
    with tf.name_scope("D_real"), tf.variable_scope("D"):
        D_real_output = GANDiscriminator(x)
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")

    # Discriminator with fake input data
    with tf.name_scope("D_fake"), tf.variable_scope("D", reuse=True):
        D_fake_output = GANDiscriminator(G_output)

    # Only use the WGAN-GP loss for now
    G_loss = -tf.reduce_mean(D_fake_output)
    D_loss = tf.reduce_mean(D_fake_output) - tf.reduce_mean(D_real_output)

    alpha = tf.random_uniform(shape=[batch_size, 1, 1], minval=0., maxval=1.)

    # Difference between real input and generator (fake) output
    differences = G_output - x

    interpolates = x + (alpha * differences)
    with tf.name_scope("D_interpolates"), tf.variable_scope("D", reuse=True):
        D_interpolates_output = GANDiscriminator(interpolates)

    LAMBDA = 10
    gradients = tf.gradients(D_interpolates_output, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)

    D_loss += LAMBDA * gradient_penalty

    # Optimisers - pretty sure for progressive growing changes might be needed, look at the PGGAN paper
    G_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    )
    D_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    )

    # Training ops
    G_train_op = G_opt.minimize(G_loss, var_list=G_vars, global_step=tf.train.get_or_create_global_step())
    D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

    # Training
    # TODO: This'll definitely have to be changed
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir="checkpoints/",
        save_checkpoint_secs=300,
        save_summaries_secs=300) as sess:
        while True:

            # Train discriminator
            for i in range(D_UPDATES_PER_G_UPDATE):
                sess.run(D_train_op)

            sess.run(G_train_op)

