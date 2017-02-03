# IMPORTANT:
# the IDs given with TRAIN and TEST variables (ff1010bird, warblrb10k, testdata, etc.)
# must correspond to
# - label files (files must be named $LABELPATH/$ID.csv)
# - audio files (files must be $AUDIOPATH/$ID/*.wav)

# path where label files are found
LABELPATH="/home/thomas/salsa/bird_challenge/labels/"
# path where audio files are found (in dataset subfolders)
AUDIOPATH="/home/thomas/salsa/bird_challenge/audio/"
# datasets making up the training set
TRAIN="ff1010bird warblrb10k"
# dataset making up the testing set (only one!)
TEST=testdata

# where to put work data (must be writable)
WORKPATH="/tmp/bird_workspace/"
WORKPATH="/media/audiostreams/data/bird-challenge/results/master/"

# network configuration to use (network_$NETWORK.inc file)
NETWORK=final_submission

# email for notification on finished subtasks (if email address is given)
EMAIL=
