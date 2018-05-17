Bird audio detection challenge 2018 - DCASE Task 3
==================================================

This is a bird audio detection system, derived from [Thomas Grill's "bulbul" system](https://jobim.ofai.at/gitlab/gr/bird_audio_detection_challenge_2017/tree/master), and modified to work as a baseline for the 2018 DCASE Task 3 [Bird Audio Detection task](http://dcase.community/challenge2018/task-bird-audio-detection).

To use the system with the DCASE 2018 data, ensure that the WAV and CSV data files are arranged in the following subfolders (you may need to rename the downloaded files):

* audio/
     * BirdVox-DCASE-20k/
     * ff1010bird/
     * warblrb10k/
* labels/
     * BirdVox-DCASE-20k.csv
     * ff1010bird.csv
     * warblrb10k.csv

The neural network system is desribed in the following publication:

Grill, T. & Schlüter, J. (2017) Two Convolutional Neural Networks for Bird Detection in Audio Signals. 25th European Signal Processing Conference (EUSIPCO2017). Kos, Greece.
  https://doi.org/10.23919/EUSIPCO.2017.8081512

The system includes the ability to run in two stages, with 'pseudo-labelling' added after the first stage. For the baseline we only use the first stage.

We have also modified the script so that the 3 training sets are used as the basis for the 3-fold crossvalidation used during training and validation, as recommended for the 2018 task.

Thomas Grill's original readme is below:

 = = =


Bird audio detection challenge 2017
===================================

This is the code base for the [bird audio detection challenge 2017](http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/) using convolutional neural networks (CNNs) working on spectrograms.
The winning submission is tagged ['official_submission'](https://jobim.ofai.at/gitlab/gr/bird_audio_detection_challenge_2017/tree/official_submission).

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
Detailed installation instructions for Theano and lasagne can be found on https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne .

simplenn: https://jobim.ofai.at/gitlab/gr/simplenn
For the 'official_submission' a specially tagged version of simplenn should be installed:
https://jobim.ofai.at/gitlab/gr/simplenn/repository/archive?ref=bird_audio_detection_challenge_2017


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

The spectrograms are calculated on audio that is resampled to 22k sample rate using ffmpeg/avconv before the STFT computation. It has been found that the method/quality of resampling varies greatly across different ffmpeg/avconv versions.
Our best results were achieved using avconv version 9.20-6:9.20-0ubuntu0.14.04.1 which employs an anti-aliasing low-pass filter with a relatively shallow slope prior to resampling.
The performance differences to "better" resampling implementations using a steep anti-aliasing filter are noticeable and still subject to investigation.
A portable (but otherwise identical) variation avoiding the use of ffmpeg/avconv for WAV files with 44.1 kHz sample rate can be found with tag ['portable_submission'](https://jobim.ofai.at/gitlab/gr/bird_audio_detection_challenge_2017/tree/portable_submission). As stated above, the performance is slightly lower (about 1% AUROC) than the best results.
