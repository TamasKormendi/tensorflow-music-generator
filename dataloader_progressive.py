import tensorflow as tf
import numpy as np
import utils
import glob
import os

# Positive and negative range of a 16-bit signed int
# with this we can scale the data to [-1, 1] inclusive range
BIT_RANGE = 32767

class Dataloader(object):

    def __init__(self, window_length, batch_size, filepath, num_channels, augmentation_level=0):
        """
        :param window_length: the amount of samples passed to the 1st conv layer
        :param batch_size: the amount of desired batches
        :param filepath: directory path to the directory where training data is stored
        """
        self.num_channels = num_channels
        self.window_length = window_length
        self.batch_size = batch_size

        self.sampling_rate, self.all_sliced_samples = self.process_directory(filepath, augmentation_level=augmentation_level)

        # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python/17511341#17511341
        # Basically math.ceil() but with support for big ints
        self.num_batches = -(-len(self.all_sliced_samples) // batch_size)

    def process_file(self, filepath):
        """
        Load a 16-bit PCM wav file and preprocess it:
            Scale values to [-1, 1] - inclusive
            :return: Sampling rate and a list containing the data
        """

        sampling_rate, raw_samples = utils.read_wav_file(filepath)

        scaled_samples = np.array(raw_samples)

        scaled_samples.reshape(-1, self.num_channels)
        scaled_samples.shape = (len(scaled_samples), self.num_channels)

        print("File loaded into memory")

        scaled_samples = scaled_samples.astype(np.float32) / BIT_RANGE

        print("File at {} loaded".format(filepath))

        print(scaled_samples.shape)

        return sampling_rate, scaled_samples

    def process_directory(self, directory_path, augmentation_level=0):
        """
        Load all the wav files in a directory, pad them to be divisible by window_length, \n
        slice them up into window_length chunks and return them as a numpy array
        :param directory_path: the path to the directory where the WAV files are
        :param augmentation_level: specify the amount of data augmentation. Only use with very small datasets
        :return: the sampling rate and all the data as a sliced (window_length chunks) numpy array
        """
        all_samples = np.array([])
        all_samples.shape = (0, self.num_channels)
        print(all_samples.shape)

        sliced_samples = []
        sampling_rate = 0

        # All the wav files should have the same sampling rate
        for filename in glob.glob(os.path.join(directory_path, "*.wav")):
            sampling_rate, current_samples = self.process_file(filename)

            all_samples = np.concatenate((all_samples, current_samples))

        print(all_samples.shape)

        assert (len(all_samples) != 0), "No training data provided"

        # Data augmentation: offsets the start of samples by win_length // 10 and appends it to the
        # end of the all_samples list - can be very memory expensive so use it only for small datasets
        if augmentation_level > 0:
            assert (len(all_samples) > self.window_length), "Data augmentation is switched on but there are fewer samples than the window length"
            assert augmentation_level < 10, "Data augmentation level should be below 10, it was {}".format(augmentation_level)

            augmented_data_start = self.window_length // 10
            augmented_data_end = len(all_samples)

            for i in range(augmentation_level):
                all_samples = np.concatenate((all_samples, all_samples[augmented_data_start:augmented_data_end]))

                augmented_data_start += self.window_length // 10

        # Pad our all_samples array so it is divisible by window_length
        if len(all_samples) % self.window_length != 0:
            remainder = len(all_samples) % self.window_length

            padding_length = self.window_length - remainder
            if self.num_channels == 1:
                pad_list = [0] * padding_length
                pad_list = np.array(pad_list)
                pad_list.shape = (padding_length, 1)
                all_samples = np.concatenate((all_samples, pad_list))
            else:
                all_samples = np.concatenate((all_samples, [[0, 0]] * padding_length))

        # Slice all the data into window_length chunks so they can be batched later
        index = 0

        prev_slice_length = 0
        current_slice_length = 0
        counter = 0
        mismatch = False

        while index < len(all_samples):
            if mismatch:
                print("Not the last value")

            current_slice = all_samples[index:index + self.window_length]

            if counter == 0:
                prev_slice_length = len(current_slice)
                current_slice_length = len(current_slice)
                counter += 1
            else:
                current_slice_length = len(current_slice)

            if current_slice_length != prev_slice_length:
                print("Slice length mismatch, previous: {}, current: {}".format(prev_slice_length, current_slice_length))
                mismatch = True

            current_slice_reshaped = np.asarray(current_slice, dtype=np.float32)
            # Second argument is the channel amount
            current_slice_reshaped.shape = (self.window_length, self.num_channels)

            sliced_samples.append(current_slice_reshaped)

            index += self.window_length

            prev_slice_length = len(current_slice)

        print("All files loaded")

        return sampling_rate, np.asarray(sliced_samples, dtype=np.float32)

    def get_next(self):
        """
        Get the next window_size samples and return them to be used in an input feed_dict (for now)
        In the future might move to a batched implementation
        :return: Return the next window_size samples
        """

        data_placeholder = tf.placeholder(self.all_sliced_samples.dtype, self.all_sliced_samples.shape, name="data")

        # Create a dataset and batch it
        # dataset = tf.data.Dataset.from_tensor_slices(self.all_sliced_samples)
        dataset = tf.data.Dataset.from_tensor_slices(data_placeholder)

        dataset = dataset.shuffle(buffer_size=4096)

        # If (self.batch_size, True) the last batch gets dropped if size < normal batch_size
        # Current implementation is way too reliant on fixed batch sizes so the remainder is dropped
        dataset = dataset.batch(self.batch_size, True)

        # TODO: has to be changed if the training gets split to separate epochs, need TODO below too for that
        dataset = dataset.repeat()
        # TODO: Might have to change this to an initialisable iterator if we run into memory issues
        iterator = dataset.make_initializable_iterator()

        return iterator