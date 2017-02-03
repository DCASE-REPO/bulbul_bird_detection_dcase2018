Bird audio detection challenge 2017
===================================

This is a submission for the [bird audio detection challenge 2017](http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/) using convolutional neural networks (CNNs) working on spectrograms.

Contact address: Thomas Grill (thomas.grill@ofai.at)


Description:
------------

In a first run, the networks train on the whole provided training data and make predictions on the testing data. "Safe" predictions (close to 0 or 1) are then added to the training data as so-called "pseudo-labeled" data. A second run in performed on the extended training set. Afterwards, all predictions are bagged to yield final predictions.

The implementation is done in the Python programming language using numpy, Theano and Lasagne packages, as well as the custom lasagne front-end simplenn. All softwares used are open source and cross-platform. In order for the CNN training to run at acceptable speeds, a GPU is required. Memory requirements are about 6 GiB CPU RAM, and 1.5 GiB GPU RAM.


Required software:
------------------

Python (version 2.7): https://www.python.org/
numpy: http://www.numpy.org
Theano: https://github.com/Theano/Theano
lasagne: https://github.com/Lasagne/Lasagne
simplenn: https://jobim.ofai.at/gitlab/gr/simplenn (git clone might not work, use archive download)

Detailed installation instructions for Theano and lasagne can be found on https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne .


Running the training/prediction:
--------------------------------

1. Adjust the paths to labels and audiofiles, as well as other basic settings in **config.inc**.

2. Run the training/prediction procedure by executing **run.sh**.


With **run.sh**, several steps are executed in sequence:

* **stage1_prepare**: Generation of training and testing data filelists. Generation of spectrograms for the audio files.

* **stage1_train**: First training run, producing network models

* **stage1_predict**: Predictions based on these models

* **stage2_prepare**: Generating pseudo-labeled additional training data

* **stage2_train**: Second training run, producing more network models

* **stage2_predict**: Final predictions, employing all network models


If the spectrogram files, network models or prediction files are already present, they are not regenerated.


Advanced use:
-------------

Each of the steps executed in **run.sh** can be run explicitly by specifying them as the first argument, such as in **run.sh stage1_train**.
For the training steps, model indices can also be specified, e.g., **run.sh stage1_train 1**, with the index running from 1 to the number of models (typically 5).
This can be used to train models in parallel, on several GPUs (or CPU cores).


Important note:
---------------

The spectrograms are calculated on audio that is resampled to 22k sample rate using ffmpeg/avconv before the STFT computation. It has been found that the method/quality of resampling varies greatly across different versions.
Our best results were achieved using avconv version 9.20-6:9.20-0ubuntu0.14.04.1 which employs an anti-aliasing low-pass filter with a relatively shallow slope prior to resampling.
The performance differences to "better" resampling implementations using a steep anti-aliasing filter are noticeable and still subject to investigation.
