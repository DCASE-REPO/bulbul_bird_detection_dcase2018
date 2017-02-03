#!/bin/bash
here="${0%/*}"

AUDIO=$1
SPECT=$2
SR=${3:-22050}
FPS=${4:-70}
FFTLEN=${5:-1024}
FMIN=${6:-50}
FMAX=${7:-11000}
BANDS=${8:-80}

for f in ${AUDIO}/*/*.wav
do
    b="${f##*/}"
    p="${f%/*}"  # full path
    sp="${p##*/}"  # subpath
    mkdir $SPECT/$sp 2> /dev/null
    o="$SPECT/$sp/${b}.h5"
    if [ ! -f "$o" ]; then
        echo "Making ${o}"
        if ! $here/extract_melspect.py --channels=mix-after -r ${SR} -f ${FPS} -l ${FFTLEN} -t mel -m ${FMIN} -M ${FMAX} -b ${BANDS} -s log --featname "features" --include-times --times-mode=borders "$f" "$o"; then
            echo "Failed making ${o} - exiting"
            exit $?
        fi
    fi
done
