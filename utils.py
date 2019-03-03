import numpy as np
import scipy.io.wavfile
import wave
import glob
import os

#Note: does not read 24-bit files
def read_wav_file(filepath):
    sampling_rate, samples = scipy.io.wavfile.read(filepath)

    return sampling_rate, samples

def write_wav_file(filepath, sample_rate, samples):
    scipy.io.wavfile.write(filepath, sample_rate, samples)

# Only check one file within the directory
# It is assumed every file has the same amount of channels in the dir
def get_num_channels(data_directory):
    channel_count = 0
    for filename in glob.glob(os.path.join(data_directory, "*.wav")):
        open_file = wave.open(filename, "r")
        channel_count = open_file.getnchannels()
        open_file.close()
        break

    assert (channel_count > 0 and channel_count <3), "Channel count has to be 1 or 2, it was {}".format(channel_count)
    return channel_count

"""
if __name__:
    #Do not do this for normal usage but for testing it is fine
    np.set_printoptions(threshold = np.nan)

    sample_rate, samples = read_wav_file("../test.wav")

    print(sample_rate)

    print(samples)

    write_wav_file("../test_write.wav", sample_rate, samples)
"""