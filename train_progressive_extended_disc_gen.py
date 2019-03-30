"""
This file is based on WaveGAN v1: https://github.com/chrisdonahue/wavegan/tree/v1
and the Tensorflow Models implementation of PGGAN: https://github.com/tensorflow/models/tree/master/research/gan/progressive_gan

Both of these are heavily modified so it is not practical to point out which section of code is inspired by which, but in this file
the train(), infer() and preview() functions are mostly adapted from WaveGAN, while the checkpointing functionality is adapted from
PGGAN.

Mixed precision training was inspired by these slides: http://on-demand.gputechconf.com/gtc-taiwan/2018/pdf/5-1_Internal%20Speaker_Michael%20Carilli_PDF%20For%20Sharing.pdf

If code is adapted from other sources it is explicitly pointed out.
"""

import os
import time

import tensorflow as tf
import pickle

from model_progressive_extended_disc_gen import GANGenerator, GANDiscriminator, block_name
import dataloader_progressive as dataloader
import utils

import argparse


SAMPLING_RATE = 16000
# 100 random inputs for the generator
G_INIT_INPUT_SIZE = 100
D_UPDATES_PER_G_UPDATE = 5

# Set later in main properly
window_size = 16384
# Default batch size - set in main properly
batch_size = 64

# G = generator
# D = discriminator

def train(training_data_dir, train_dir, stage_id, num_channels, freeze_early_layers=False, use_mixed_precision_training = False, augmentation_level=0):
    print("Training called")

    loader = dataloader.Dataloader(window_size, batch_size, training_data_dir, num_channels, augmentation_level=augmentation_level)

    iterator = loader.get_next()

    # Input (real) training data
    x = iterator.get_next()

    print(x.get_shape())

    G_input = tf.random_uniform([batch_size, G_INIT_INPUT_SIZE], -1., 1., dtype=tf.float32)

    if use_mixed_precision_training:
        # Generator network
        G_input = tf.cast(G_input, tf.float16)
        with tf.variable_scope("G", custom_getter=float32_variable_storage_getter):
            G_output = GANGenerator(G_input, train=True, num_blocks=stage_id, freeze_early_layers=freeze_early_layers, channels=num_channels)
        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G")

        # Discriminator with real input data
        x = tf.cast(x, tf.float16)
        with tf.name_scope("D_real"), tf.variable_scope("D", custom_getter=float32_variable_storage_getter):
            D_real_output = GANDiscriminator(x, num_blocks=stage_id, freeze_early_layers=freeze_early_layers)
        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")

        # Discriminator with fake input data
        # G_output is already in fp16
        with tf.name_scope("D_fake"), tf.variable_scope("D", reuse=True, custom_getter=float32_variable_storage_getter):
            D_fake_output = GANDiscriminator(G_output, num_blocks=stage_id, freeze_early_layers=freeze_early_layers)

        x = tf.cast(x, tf.float32)
        G_output = tf.cast(G_output, tf.float32)
        D_real_output = tf.cast(D_real_output, tf.float32)
        D_fake_output = tf.cast(D_fake_output, tf.float32)
    else:
        # Generator network
        with tf.variable_scope("G"):
            G_output = GANGenerator(G_input, train=True, num_blocks=stage_id, freeze_early_layers=freeze_early_layers, channels=num_channels)
        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G")

        # Discriminator with real input data
        with tf.name_scope("D_real"), tf.variable_scope("D"):
            D_real_output = GANDiscriminator(x, num_blocks=stage_id, freeze_early_layers=freeze_early_layers)
        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")

        # Discriminator with fake input data
        with tf.name_scope("D_fake"), tf.variable_scope("D", reuse=True):
            D_fake_output = GANDiscriminator(G_output, num_blocks=stage_id, freeze_early_layers=freeze_early_layers)



    # Write generator summary
    tf.summary.audio("real_input", x, SAMPLING_RATE)
    tf.summary.audio("generator_output", G_output, SAMPLING_RATE)
    # RMS = reduced mean square
    G_output_rms = tf.sqrt(tf.reduce_mean(tf.square(G_output[:, :, 0]), axis=1))
    real_input_rms = tf.sqrt(tf.reduce_mean(tf.square(x[:, :, 0]), axis=1))
    tf.summary.histogram("real_input_rms_batch", real_input_rms)
    tf.summary.histogram("G_output_rms_batch", G_output_rms)
    # Reduce the rms of batches into a single scalar
    tf.summary.scalar("real_input_rms", tf.reduce_mean(real_input_rms))
    tf.summary.scalar("G_output_rms", tf.reduce_mean(G_output_rms))

    # Only use the WGAN-GP loss for now
    G_loss = -tf.reduce_mean(D_fake_output)
    D_loss = tf.reduce_mean(D_fake_output) - tf.reduce_mean(D_real_output)

    alpha = tf.random_uniform(shape=[batch_size, 1, 1], minval=0., maxval=1.)

    # Difference between real input and generator (fake) output
    differences = G_output - x

    interpolates = x + (alpha * differences)
    LAMBDA = 10

    if use_mixed_precision_training:
        interpolates = tf.cast(interpolates, tf.float16)
        with tf.name_scope("D_interpolates"), tf.variable_scope("D", reuse=True, custom_getter=float32_variable_storage_getter):
            D_interpolates_output = GANDiscriminator(interpolates, num_blocks=stage_id, freeze_early_layers=freeze_early_layers)

        gradients = tf.gradients(D_interpolates_output, [interpolates])[0]
        gradients = tf.cast(gradients, tf.float32)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)

        D_loss += LAMBDA * gradient_penalty
    else:
        with tf.name_scope("D_interpolates"), tf.variable_scope("D", reuse=True):
            D_interpolates_output = GANDiscriminator(interpolates, num_blocks=stage_id, freeze_early_layers=freeze_early_layers)

        gradients = tf.gradients(D_interpolates_output, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)

        D_loss += LAMBDA * gradient_penalty

    tf.summary.scalar("Output/real_output", tf.reduce_mean(D_real_output))
    tf.summary.scalar("Output/fake_output", tf.reduce_mean(D_fake_output))
    tf.summary.scalar("Output/mixed_output", tf.reduce_mean(D_interpolates_output))

    tf.summary.scalar("Loss/Generator_loss", G_loss)
    tf.summary.scalar("Loss/Discriminator_loss", D_loss)

    with tf.variable_scope("optimiser_vars") as var_scope:
        # Optimisers
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

        if use_mixed_precision_training:
            # A loss scale of 32 ended up being stable for both networks
            loss_scale = 32.0

            G_gradients, G_variables = zip(*G_opt.compute_gradients(G_loss * loss_scale, var_list=G_vars))
            G_gradients = [gradient / loss_scale for gradient in G_gradients]
            G_train_op = G_opt.apply_gradients(zip(G_gradients, G_variables), global_step=tf.train.get_or_create_global_step())

            loss_scale_discriminator = 32.0

            D_gradients, D_variables = zip(*D_opt.compute_gradients(D_loss * loss_scale_discriminator, var_list=D_vars))
            D_gradients = [gradient / loss_scale_discriminator for gradient in D_gradients]
            D_train_op = D_opt.apply_gradients(zip(D_gradients, D_variables))
        else:
            # Training ops - need to specify the var_list so it does not default to all vars within TRAINABLE_VARIABLES
            # See: https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer#minimize
            G_train_op = G_opt.minimize(G_loss, var_list=G_vars, global_step=tf.train.get_or_create_global_step())
            D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

    optimiser_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=var_scope.name)

    scaffold = make_custom_scaffold(stage_id, optimiser_vars, train_dir, freeze_early_layers)

    if freeze_early_layers:
        print("Early layers frozen")
    else:
        print("Training all layers")

    iterator_init_hook = IteratorInitiasliserHook(iterator, loader.all_sliced_samples)

    # Training
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=get_train_subdirectory(stage_id, train_dir, freeze_early_layers),
        save_checkpoint_secs=300,
        save_summaries_secs=120,
        hooks=[iterator_init_hook],
        scaffold=scaffold) as sess:
        print("Training start")
        while True:

            # Train discriminator
            for i in range(D_UPDATES_PER_G_UPDATE):
                sess.run(D_train_op)

            sess.run(G_train_op)

def infer(train_dir, stage_id, num_channels, use_mixed_precision_training=False):
    infer_dir = os.path.join(train_dir, "infer")
    if not os.path.isdir(infer_dir):
        os.makedirs(infer_dir)

    # Subgraph that generates latent vectors

    # Number of samples to generate
    sample_amount = tf.placeholder(tf.int32, [], name="samp_z_n")
    # Input samples
    input_samples = tf.random_uniform([sample_amount, G_INIT_INPUT_SIZE], -1.0, 1.0, dtype=tf.float32, name="samp_z")

    input_placeholder = tf.placeholder(tf.float32, [None, G_INIT_INPUT_SIZE], name="z")
    flat_pad = tf.placeholder(tf.int32, [], name="flat_pad")

    # Run the generator
    if use_mixed_precision_training:
        with tf.variable_scope("G", custom_getter=float32_variable_storage_getter):
            generator_output = GANGenerator(input_placeholder, train=False, num_blocks=stage_id, channels=num_channels)
    else:
        with tf.variable_scope("G"):
            generator_output = GANGenerator(input_placeholder, train=False, num_blocks=stage_id, channels=num_channels)
    generator_output = tf.identity(generator_output, name="G_z")

    # Flatten batch and pad it so there is a pause between generated samples
    # Only generate one file
    num_channels = int(generator_output.get_shape()[-1])
    output_padded = tf.pad(generator_output, [[0, 0], [0, flat_pad], [0, 0]])
    output_flattened = tf.reshape(output_padded, [-1, num_channels], name="G_z_flat")

    # Encode to int16 - assumes division by 32767 to encode to [-1, 1] float range
    def float_to_int16(input_values, name=None):
        input_int16 = input_values * 32767
        input_int16 = tf.clip_by_value(input_int16, -32767., 32767)
        input_int16 = tf.cast(input_int16, tf.int16, name=name)
        return input_int16
    generator_output_int16 = float_to_int16(generator_output, name="G_z_int16")
    generator_output_flat_int16 = float_to_int16(output_flattened, name="G_z_flat_int16")

    # Create saver
    G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="G")
    # Need to scope the global_step to the same scope that it has in the train function
    with tf.variable_scope("optimiser_vars") as var_scope:
        global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver(G_vars + [global_step])

    # Export graph
    tf.train.write_graph(tf.get_default_graph(), infer_dir, "infer.pbtxt")

    # Export metagraph
    infer_metagraph_filepath = os.path.join(infer_dir, "infer.meta")
    tf.train.export_meta_graph(
        filename=infer_metagraph_filepath,
        clear_devices=True,
        saver_def=saver.as_saver_def()
    )

    tf.reset_default_graph()

    print("Metagraph construction done")

# Generate preview audio files - should be run in a separate process, parallel to or after training
# Might be problems with it when saving random input vectors with a given amount_to_preview
# and using another value when running it again
def preview(train_dir, amount_to_preview):
    preview_dir = os.path.join(train_dir, "preview")
    if not os.path.isdir(preview_dir):
        os.makedirs(preview_dir)

    # Graph loading - might or might not have to change this, we'll see
    infer_metagraph_filepath = os.path.join(train_dir, "infer", "infer.meta")
    graph = tf.get_default_graph()
    saver = tf.train.import_meta_graph(infer_metagraph_filepath)

    # Generate or restore the input random latent vector for the generator
    # For now restoration is commented out - generate a new latent vector for every run
    # It makes sense to use a fixed latent vector while actively training so the improvement
    # can be heard

    # input_filepath = os.path.join(preview_dir, "z.pkl")
    # if os.path.exists(input_filepath):
    #     with open(input_filepath, "rb") as f:
    #         input_values = pickle.load(f)
    # else:


    # Generate random input values for the generator
    sample_feeds = {}
    sample_feeds[graph.get_tensor_by_name("samp_z_n:0")] = amount_to_preview
    sample_fetches = {}
    # "zs" are the random input values
    sample_fetches["zs"] = graph.get_tensor_by_name("samp_z:0")
    with tf.Session() as sess:
        fetched_values = sess.run(sample_fetches, sample_feeds)
    input_values = fetched_values["zs"]


    # Save random input
    # with open(input_filepath, "wb") as f:
    #   pickle.dump(input_values, f)


    # Set up the graph for the generator
    feeds = {}
    feeds[graph.get_tensor_by_name("z:0")] = input_values
    # Leave 1 second of no audio between samples
    feeds[graph.get_tensor_by_name("flat_pad:0")] = SAMPLING_RATE
    fetches = {}
    fetches["step"] = tf.train.get_or_create_global_step()
    # Output of the generator
    fetches["G_z"] = graph.get_tensor_by_name("G_z:0")
    # Output of the generator, flattened and transformed to int16
    fetches["G_z_flat_int16"] = graph.get_tensor_by_name("G_z_flat_int16:0")

    # Write summary
    output = graph.get_tensor_by_name("G_z_flat:0")
    summaries = [tf.summary.audio("preview", tf.expand_dims(output, axis=0), SAMPLING_RATE, max_outputs=1)]
    fetches["summaries"] = tf.summary.merge(summaries)
    summary_writer = tf.summary.FileWriter(preview_dir)

    # Loop, wait until a new checkpoint is found - if found, execute
    checkpoint_filepath = None
    while True:
        latest_checkpoint_filepath = tf.train.latest_checkpoint(train_dir)

        if latest_checkpoint_filepath != checkpoint_filepath:
            print("Preview: {}".format(latest_checkpoint_filepath))

            with tf.Session() as sess:
                saver.restore(sess, latest_checkpoint_filepath)

                fetches_results = sess.run(fetches, feeds)
                training_step = fetches_results["step"]

            preview_filepath = os.path.join(preview_dir, "{}.wav".format(str(training_step).zfill(8)))
            utils.write_wav_file(preview_filepath, SAMPLING_RATE, fetches_results["G_z_flat_int16"])

            summary_writer.add_summary(fetches_results["summaries"], training_step)

            print("Wav written")

            checkpoint_filepath = latest_checkpoint_filepath

        time.sleep(1)


# Stage_id is equivalent to num_blocks
def make_custom_scaffold(stage_id, optimiser_var_list, training_root_directory, early_layers_frozen):
    restore_var_list = []
    previous_checkpoint = None
    current_checkpoint = tf.train.latest_checkpoint(get_train_subdirectory(stage_id, training_root_directory, early_layers_frozen))
    # Skip var restoration if training only has 1 block and no saved checkpoint
    if stage_id > 1 and current_checkpoint is None:

        # Check if a frozen checkpoint exists from the current level
        previous_checkpoint = tf.train.latest_checkpoint(get_train_subdirectory(stage_id, training_root_directory, True))
        # If yes, restore every non-optimiser variable
        if previous_checkpoint is not None:
            restore_var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                                if var not in set(optimiser_var_list)]
        # If not, check if a non-frozen one exists from the previous level and restore
        # every non-optimiser variable that are from the old blocks
        else:
            previous_checkpoint = tf.train.latest_checkpoint( get_train_subdirectory(stage_id - 1, training_root_directory, False))

            number_of_blocks = stage_id
            prev_num_blocks = stage_id - 1

            new_block_var_list = []
            for block_id in range(prev_num_blocks + 1, number_of_blocks + 1):
                new_block_var_list.extend(
                    tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope=".*/{}/".format(block_name(block_id))
                    )
                )
                new_block_var_list.extend(
                    tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope=".*{}_output/".format(block_name(block_id))
                    )
                )
                new_block_var_list.extend(
                    tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope=".*{}_input/".format(block_name(block_id))
                    )
                )

            restore_var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                                if var not in set(optimiser_var_list + new_block_var_list)]
    elif current_checkpoint is not None:
        restore_var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]

    saver_for_restoration = tf.train.Saver(var_list=restore_var_list, allow_empty=True)

    init_op = tf.global_variables_initializer()

    def initialisation_function(unused_scaffold, sess):
        sess.run(init_op)
        print("Variables to restore:")
        print("\n".join([var.name for var in restore_var_list]))
        if current_checkpoint is not None:
            print("Restoring variables from current checkpoint")
            saver_for_restoration.restore(sess, current_checkpoint)
        elif previous_checkpoint is not None:
            print("Restoring variables from previous checkpoint")
            saver_for_restoration.restore(sess, previous_checkpoint)

    # Init_op is dummy since init_fn does the initialisation
    return tf.train.Scaffold(init_op=tf.constant([]), init_fn=initialisation_function)

# Return the name of a possibly existing training subdirectory
# Stage_id corresponds to num_blocks
def get_train_subdirectory(stage_id, training_root_directory, early_layers_frozen):
    if early_layers_frozen:
        return os.path.join(training_root_directory, "stage_{:05d}_frozen".format(stage_id))
    else:
        return os.path.join(training_root_directory, "stage_{:05d}".format(stage_id))

# Base exponent is 4 since the length of the output from the 1st conv layer is 2^6
def get_window_length(num_blocks):
    base_exponent = 4
    block_multiplier = 2 * num_blocks
    return 2 ** (base_exponent + block_multiplier)

# Adapted from https://github.com/tensorflow/tensorflow/issues/12859
class IteratorInitiasliserHook(tf.train.SessionRunHook):
    def __init__(self, iterator, data):
        self.iterator = iterator
        self.data = data
    def begin(self):
        self.initialiser = self.iterator.initializer
    def after_create_session(self, session, coord):
        del coord
        session.run(self.initialiser, feed_dict={"data:0": self.data})

# Adapted from http://on-demand.gputechconf.com/gtc-taiwan/2018/pdf/5-1_Internal%20Speaker_Michael%20Carilli_PDF%20For%20Sharing.pdf
def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
  """Custom variable getter that forces trainable variables to be stored in
  float32 precision and then casts them to the training precision.
  """
  storage_dtype = tf.float32 if trainable else dtype
  variable = getter(name, shape, dtype=storage_dtype,
                    initializer=initializer, regularizer=regularizer,
                    trainable=trainable,
                    *args, **kwargs)
  if trainable and dtype != tf.float32:
    variable = tf.cast(variable, dtype)
  return variable

if __name__ == "__main__":

    training_data_dir = "data/"
    training_dir = "checkpoints/"
    amount_to_preview = 5

    parser = argparse.ArgumentParser()

    parser.add_argument("--preview", help="Switches on preview mode.", action="store_true")
    parser.add_argument("--num_blocks", help="Specify the number of blocks", type=int, default=5)
    parser.add_argument("--use_mixed_precision_training", help="If specified, uses mixed precision training", action="store_true")
    parser.add_argument("--augmentation_level", help="Specify the level of data augmentation. Only recommended for small datasets.", type=int, default=0)
    parser.add_argument("--use_sample_norm", help="If specified, uses sample normalisation", action="store_true")
    parser.add_argument("--freeze_early_layers", help="If specified, freezes early layers", action="store_true")
    parser.add_argument("--batch_size", help="Specify batch size. If none given it is automatically selected. If an OOM error is encountered run the training again with a lower amount.", type=int)

    args = parser.parse_args()

    preview_mode = args.preview
    num_blocks = args.num_blocks
    use_mixed_precision_training = args.use_mixed_precision_training
    augmentation_level = args.augmentation_level

    assert (num_blocks >= 1 and num_blocks < 9), "The number of blocks should be between 1 and 8 inclusive, it was {}".format(num_blocks)

    if not preview_mode:
        mode = "train"
    else:
        mode = "preview"

    window_size = get_window_length(num_blocks)
    print("Window size: {}".format(window_size))

    if num_blocks < 2:
        freeze_early_layers = False
    else:
        freeze_early_layers = args.freeze_early_layers

    channel_count = utils.get_num_channels(training_data_dir)

    suitable_batch_size_dict_high_vram = {1 : 128,
                                2 : 112,
                                3 : 96,
                                4 : 80,
                                5 : 64,
                                6 : 32,
                                7 : 8,
                                8 : 4}

    if not args.batch_size:
        batch_size = suitable_batch_size_dict_high_vram[num_blocks]
    else:
        batch_size = args.batch_size

    if mode == "train":
        infer(get_train_subdirectory(num_blocks, training_dir, freeze_early_layers), num_blocks, channel_count, use_mixed_precision_training=use_mixed_precision_training)
        train(training_data_dir, training_dir, num_blocks, channel_count, freeze_early_layers=freeze_early_layers, use_mixed_precision_training=use_mixed_precision_training, augmentation_level=augmentation_level)
    elif mode == "preview":
        preview(get_train_subdirectory(num_blocks, training_dir, freeze_early_layers), amount_to_preview)
    elif mode == "infer":
        infer(get_train_subdirectory(num_blocks, training_dir, freeze_early_layers), num_blocks, channel_count, use_mixed_precision_training=use_mixed_precision_training)