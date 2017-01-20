# Bird audio detection challenge 2017

This is a submission for the [bird audio detection challenge 2017](http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/) using convolutional neural networks (CNNs) working on spectrograms.

Contact address: Thomas Grill <thomas.grill@ofai.at>

## Description:

In a first run, the networks train on the whole provided training data and make predictions on the testing data. "Safe" predictions (close to 0 or 1) are then added to the training data as so-called "pseudo-labeled" data. A second run in performed on the extended training set. Afterwards, all predictions are bagged to yield final predictions.

The implementation is done in the Python programming language using numpy, Theano and Lasagne packages, as well as the custom lasagne front-end simplenn. All softwares used are open source and cross-platform. In order for the CNN training to run at acceptable speeds, a GPU is required. Consult the Theano docs for details. Memory requirements are about 6 GiB CPU RAM, and 3 GiB GPU RAM.

## Required software:

Python (version 2.7): https://www.python.org/
numpy: http://www.numpy.org/
Theano: https://github.com/Theano/Theano
lasagne: https://github.com/Lasagne/Lasagne
simplenn: https://jobim.ofai.at/gitlab/gr/simplenn

## Running the training/prediction:

1. Adjust the paths to labels and audiofiles, as well as other basic settings in config.inc .

2. Run the training/prediction procedure by executing run.sh .

With run.sh, several steps are executed in sequence:
* Generation of training and testing data filelists
* Generation of spectrograms for the audio files
* First training run, producing network models
* Predictions based on these models
* Generating pseudo-labeled additional training data
* Second training run, producing more network models
* Final predictions, emplying all network models

If the spectrogram files, network models or prediction files are already present, they are not regenerated.
