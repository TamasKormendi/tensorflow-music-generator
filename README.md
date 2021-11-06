# Adversarial music generation

Written between October 2018 and April 2019, as my master's project at King's College London. The purpose of this project was to generate raw/pcm audio using generative adversarial networks. As far as could be told, this project achieved state of the art results in terms of receptive field in early 2019, even though since then this has most likely been surpassed. A few then-novel networks are also presented aside from the main progressive GAN. This is the full source code for the project as well as the full dissertation. Please see the latter for more information and for miscellaneous features (mixed-precision training, data augmentation, etc.).

# Setup guide

The program was tested on Window 7/10 and Ubuntu 18.04. The easiest way to set up the
environment the program needs is by using Anaconda: https://www.anaconda.com/.
Depending on the hardware of the computer (i.e. has an Nvidia GPU that was released
approximately in the last 5 years or not) it has to be set up differently. CPU-only operation
is slow but easy to set up. Simply install Anaconda, make it sure it is working by using the
command:

```
conda env list
```

It should list the available Anaconda environments. If it is working, create a new Anaconda
environment with v1.12.0 of Tensorflow and Python 3.6:

```
conda create -n tf-cpu python=3.6 tensorflow=1.12.0
```

Once it is installed, activate the environment:

```
conda activate tf-cpu
```

...and done. It is a bit more complicated if a GPU is present. Make it sure that the installed
Nvidia driver version is at least v410, on Ubuntu this can be checked by inspecting the output
of nvidia-smi. If that is done, the rest of the procedure is similar. After Anaconda is installed and working, install and activate the GPU variant of Tensorflow:

```
conda create -n tf-gpu python=3.6 tensorflow-gpu=1.12.0
conda activate tf-gpu
```

That is it for the setup. In terms of system requirements, for training it is advised to have
the strongest hardware possible, especially on the GPU side, with at least 32GB main system
RAM for larger datasets. For generation (if a trained model is present) more modest hardware
is enough. In terms of OS compatibility, Windows 7/10 or Ubuntu 18.04 is recommended. It
might work on Mac OS X but it is not guaranteed.

After this is done, make it sure a ”checkpoints” and a ”data” folder are present in the folder
where the .py files are.

# Running instructions

For training, the program explicitly expects that all files are mono or stereo (only one type
should be present in the training set at the same time), 16-bit, 16 kHz audio files in .wav
format. If your training data is different, edit them with an audio editor, for example Audacity:
https://www.audacityteam.org/. Put these files in the ”data” folder. The program that shall
be run for normal usage is ”train_progressive.py”. There are a few command line options:

* –preview: instead of training, generate 5 audio slices from the latest checkpoint. It
writes a .wav file into the checkpoint folder of the chosen training stage. It also waits for
new checkpoints continuously, so exit it with CTRL+C if only one file is desired to be
generated.

* –num_blocks followed by a number between 1 and 8, inclusive: Defines how many convolutional blocks the model should have. It defaults to 5. Also known as ”stage number

* –use_mixed_precision_training: If specified, it uses mixed precision training, which is faster
on cutting-edge GPUs (Nvidia Volta or Turing) but might be slower on old GPUs.

* –augmentation_level followed by a number between 1 and 9, inclusive: It switches on
different amounts of data augmentation. Only recommended if the training set is small
(below 1 hour).

* –use_sample_norm: switches on sample normalisation. Useful for research purposes, otherwise not recommended.

* –freeze_early_layers: freezes the early layers of the model so only the new layers are trained.
Useful for progressive growing. The ideal training workflow is if training is started at stage
5: train stage 5 - train FROZEN stage 6 - train stage 6 and so on.

* –batch_size: the amount of slices the model trains on at the same time. Lower it if the
program crashes with an out-of-memory error, increase it if the GPU is under-utilised,
otherwise defaults to the following values. The first number represents the number of
blocks (stage number), the second one is the batch size for that stage:

  * 1: 128
  * 2: 112
  * 3: 96
  * 4: 80
  * 5: 64
  * 6: 32
  * 7: 8
  * 8: 4

Monitoring the training can be done through Tensorboard. Launch the training script, open
a new terminal/command line window, activate the Tensorflow environment, navigate to the
”checkpoints” directory then execute:

```
tensorboard --logdir <directory for the checkpoints of the current training stage>
```

As said previously, ”train_progressive.py” runs the stable model that has full, completely stable
functionality for both mono and stereo generation, whereas ”train.py” is stable but it is just
a modified implementation of WaveGAN, so its limitations apply. With ”train_progressive_vaegan.py”, it is advised to use Tensorboard for generating samples. The experimental models
should be stable as well but they have not been run as much as ”train_progressive.py” so do
exercise caution.

# Licence information

Unless explicitly stated otherwise, all the material in the repository is licensed under the terms of CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/legalcode. Please see the 3rd_party_copyright.txt file for complete 3rd party copyright information.

For attribution, please include the copyright, a link to the text of the copyright, my name with or without accents (Tamás Körmendi or Tamas Kormendi) and a link to this repository. Thank you.

Progressive GAN code samples are adapted from Tensorflow Models: https://github.com/tensorflow/models. The complete path for the sub-folder of PGGAN within the repository is: https://github.com/tensorflow/models/tree/master/research/gan/progressive_gan - licensed under the terms of Apache 2.0.

WaveGAN code samples are adapted from WaveGAN: https://github.com/chrisdonahue/wavegan. Especially v1 of WaveGAN, found at this tag: https://github.com/chrisdonahue/wavegan/tree/v1  - licensed under the terms of the MIT License.

VAE-GAN code samples are adapted from TF-VAEGAN: https://github.com/JeremyCCHsu/tf-vaegan - licensed under the terms of the MIT License.

My deepest thanks to everyone who directly or indirectly helped me with this project.