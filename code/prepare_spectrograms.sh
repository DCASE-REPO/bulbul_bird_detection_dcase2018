#!/bin/bash
here="${0%/*}"

AUDIO=$1
SPECT=$2

for f in ${AUDIO}/*/*.wav
do
    b="${f##*/}"
    p="${f%/*}"  # full path
    sp="${p##*/}"  # subpath
    mkdir $SPECT/$sp 2> /dev/null
    o="$SPECT/$sp/${b}.h5"
    if [ ! -f "$o" ]; then
        echo "Making ${o}"
        if ! $here/extract_melspect.py --channels=mix-after -r 22050 -f 70 -l 1024 -t mel -m 50 -M 11000 -b 80 -s log --featname "features" --include-times --times-mode=borders "$f" "$o"; then
            echo "Failed making ${o} - exiting"
            return $?
        fi
    fi
done
