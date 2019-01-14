import tensorflow as tf
import numpy as np
import utils

# One smaller than the real positive limit for 16 bits, to leave -1 and 1 available
BIT_RANGE = 32766

# TODO: get batching done and directory loading too
class Dataloader(object):

    def __init__(self, window_size, filepath):
        """
        :param window_length: the amount of samples passed to the 1st conv layer
        """
        self.window_size = window_size
        self.current_index = 0

        self.sampling_rate, self.processed_samples = self.process_file(filepath)

    def process_file(self, filepath):
        """
        Load a 16-bit PCM wav file a preprocess it:
            Scale values to (-1, 1) - exclusive\n
            Insert 1 to the beginning of the file as a beginning-of-stream token\n
            Insert -1 to the end of the file as an end-of-stream token

            :return: Sampling rate and a numpy array containing the data
        """

        sampling_rate, raw_samples = utils.read_wav_file(filepath)
        scaled_samples = []

        for sample in raw_samples:
            scaled_samples.append(sample / BIT_RANGE)

        scaled_samples.insert(0, 1)
        scaled_samples.append(-1)

        # Pad it if necessary so the length is divisible by window_size
        if len(scaled_samples) % self.window_size == 0:
            return sampling_rate, np.asarray(scaled_samples, dtype=np.float32)
        else:
            remainder = len(scaled_samples) % self.window_size

            padding_length = self.window_size - remainder
            scaled_samples.append([0] * padding_length)

            return sampling_rate, np.asarray(scaled_samples, dtype=np.float32)

    def get_next(self):
        """
        Get the next window_size samples and return them to be used in an input feed_dict (for now)
        In the future might move to a batched implementation
        :return: Return the next window_size samples
        """

        slice_to_return = self.processed_samples[self.current_index:self.current_index + self.window_size]
        self.current_index += self.window_size

        if self.current_index >= len(self.processed_samples):
            self.current_index = 0

        return slice_to_return

    def process_directory(self):
        raise NotImplemented


