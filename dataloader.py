import tensorflow as tf
import numpy as np
import utils
import glob
import os

# Positive and negative range of a 16-bit signed int
# with this we can scale the data to [-1, 1] inclusive range
BIT_RANGE = 32767

class Dataloader(object):

    def __init__(self, window_length, batch_size, filepath):
        """
        :param window_length: the amount of samples passed to the 1st conv layer
        :param batch_size: the amount of desired batches
        :param filepath: directory path to the directory where training data is stored
        """
        self.window_length = window_length
        self.batch_size = batch_size

        self.sampling_rate, self.all_sliced_samples = self.process_directory(filepath)

    def process_file(self, filepath):
        """
        Load a 16-bit PCM wav file and preprocess it:
            Scale values to [-1, 1] - inclusive
            :return: Sampling rate and a list containing the data
        """

        sampling_rate, raw_samples = utils.read_wav_file(filepath)
        scaled_samples = []

        for sample in raw_samples:
            scaled_samples.append(sample / BIT_RANGE)

        return sampling_rate, scaled_samples

    def process_directory(self, directory_path):
        """
        Load all the wav files in a directory, pad them to be divisible by window_length, \n
        slice them up into window_length chunks and return them as a numpy array
        :param directory_path: the path to the directory where the WAV files are
        :return: the sampling rate and all the data as a sliced (window_length chunks) numpy array
        """
        all_samples = []
        sliced_samples = []
        sampling_rate = 0

        # All the wav files should have the same sampling rate
        for filename in glob.glob(os.path.join(directory_path, "*.wav")):
            sampling_rate, current_samples = self.process_file(filename)

            for sample in current_samples:
                all_samples.append(sample)

        # Pad our all_samples array so it is divisible by window_length
        # Then return it as a numpy array
        assert (len(all_samples) != 0), "No training data provided"

        if len(all_samples) % self.window_length != 0:
            remainder = len(all_samples) % self.window_length

            padding_length = self.window_length - remainder
            all_samples.append([0] * padding_length)

        # Slice all the data into window_length chunks so they can be batched later
        index = 0

        while index < len(all_samples):
            current_slice = all_samples[index:index + self.window_length]
            sliced_samples.append(current_slice)

            index += self.window_length

        return sampling_rate, np.asarray(sliced_samples, dtype=np.float32)

    def get_next(self):
        """
        Get the next window_size samples and return them to be used in an input feed_dict (for now)
        In the future might move to a batched implementation
        :return: Return the next window_size samples
        """

        # Create a dataset and batch it
        dataset = tf.data.Dataset.from_tensor_slices(self.all_sliced_samples)
        # There should be no remainders, but just in case
        dataset.batch(self.batch_size, True)

        # TODO: Might have to change this to an initialisable iterator if we run into memory issues
        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()