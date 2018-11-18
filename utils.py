import numpy as np
import scipy.io.wavfile

#Note: does not read 24-bit files
def read_wav_file(filepath):
    sample_rate, samples = scipy.io.wavfile.read(filepath)

    return sample_rate, samples

def write_wav_file(filepath, sample_rate, samples):
    scipy.io.wavfile.write(filepath, sample_rate, samples)

if __name__:
    #Do not do this for normal usage but for testing it is fine
    np.set_printoptions(threshold = np.nan)

    sample_rate, samples = read_wav_file("../test.wav")

    print(sample_rate)

    print(samples)

    write_wav_file("../test_write.wav", sample_rate, samples)