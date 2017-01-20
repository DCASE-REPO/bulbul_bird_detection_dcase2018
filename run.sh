#!/bin/bash

# Training and evaluation for bird challenge 
# Thomas Grill <thomas.grill@ofai.at>
#
# Training: 6 GiB RAM, 3 GiB GPU RAM
# Evaluation: 6 GiB RAM, 1.5 GiB GPU RAM


here="${0%/*}"
cmdargs="${@:1}"

. "$here/config.inc"

# import network/learning configuration
. "$here/network_${NETWORK}.inc"


#############################
# prepare file lists
#############################

LISTPATH="$WORKPATH/filelists"

echo "Preparing file lists."
mkdir $LISTPATH 2> /dev/null
"$here/code/create_filelists.py" "$LABELPATH" $TRAIN > "$LISTPATH/train"
"$here/code/create_filelists.py" "$LABELPATH" $TEST > "$LISTPATH/test"


#############################
# prepare spectrograms
#############################

SPECTPATH="$WORKPATH/spect"

echo "Preparing spectrograms."
mkdir $SPECTPATH 2> /dev/null
"$here/code/prepare_spectrograms.sh" "${AUDIOPATH}" "${SPECTPATH}"


#############################
# define training
#############################

function train_model {
    model="$1"  # model including path
    filelists="$2"  # file list to use
    seed=$3
    
    echo "Computing model ${model} with network ${NETWORK}."

    "$here/code/simplenn_main.py" \
    --mode=train \
    --problem=binary \
    --var measures= \
    --inputs filelist:filelist \
    --var filelist:path="$LISTPATH" \
    --var filelist:lists="${filelists}" \
    --process "filelistshuffle:shuffle(seed=$seed,memory=25000)" \
    --process "input:${here}/code/load_data.py(type=spect,downmix=0,cycle=0,denoise=1,width=${net_width},seed=$seed)" \
    --var input:labels="${LABELPATH}"/'*.csv' \
    --var input:data="${SPECTPATH}/%(id)s.h5" \
    --var input:data_vars=1k \
    --process collect:collect \
    --var "collect:source=0..1"  \
    --process "scale@1:range(out_min=0.01,out_max=0.99)" \
    --layers "${net_layers}" \
    --save "${model}.h5" \
    ${net_options} \
    ${cmdargs}
}


#############################
# define evaluation
#############################

function evaluate_model {
    model="$1"  # model including path
    filelists="$2"  # file list to use
    predictions="$3"  # model including path

    echo "Evaluating model ${model}."

    "$here/code/simplenn_main.py" \
    --mode=evaluate \
    --var input:labels="${LABELPATH}"/'*.csv' \
    --var input:data="${SPECTPATH}/%(id)s.h5" \
    --var filelist:path="$LISTPATH" \
    --var filelist:lists=$filelists \
    --var filelistshuffle:bypass=1 \
    --var augment:bypass=1 \
    --load "${model}.h5" \
    --save "${predictions}.h5" \
    ${cmdargs}
}


#############################
# first run
#############################

echo "First training run."
for i in `seq $model_count`; do
    model="$WORKPATH/model_first_${i}"
    if [ ! -f "${model}.h5" ]; then # check for existence
        train_model "${model}" train $i
    else
        echo "Using existing model ${model}."
    fi
    prediction="${model}.prediction"
    if [ ! -f "${prediction}.h5" ]; then # check for existence
        evaluate_model "${model}" test "${prediction}"
    else
        echo "Using existing predictions ${prediction}."
    fi
done


#############################
# compute pseudo_labels
#############################

echo "Analyzing first run."
# prediction by bagging
"$here/code/predict.py" "$WORKPATH"/model_first_?.prediction.h5 --filelist "$LABELPATH/$TEST.csv" --filelist-header --out-prefix="$TEST/" --out-suffix='.wav' --out "$LISTPATH/test_pseudo"

# filter list by threshold
# split in half randomly
"$here/code/make_pseudo.py" "$LISTPATH/test_pseudo" ${pseudo_threshold} "$LISTPATH/test_pseudo_1" "$LISTPATH/test_pseudo_2"

# merge train filelist and half pseudo filelists
for h in 1 2; do
    cat "$LISTPATH/train" "$LISTPATH/test_pseudo_${h}" > "$LISTPATH/train_pseudo_${h}"
done


#############################
# second run
#############################

echo "Second training run."
for i in `seq $model_count`; do
    for h in 1 2; do
        model="$WORKPATH/model_second_${i}_${h}"
        if [ ! -f "${model}.h5" ]; then # check for existence
            train_model "${model}" "train_pseudo_${h}" $i
        else
            echo "Using existing model ${model}."
        fi
        prediction="${model}.prediction"
        if [ ! -f "${prediction}.h5" ]; then # check for existence
            evaluate_model "${model}" test "${prediction}"
        else
            echo "Using existing predictions ${prediction}."
        fi
    done
done


#############################################
# prediction by bagging all available models
############################################

echo "Computing final predictions."
final_predictions="$WORKPATH/prediction.csv"
"$here/code/predict.py" "$WORKPATH"/model_*.prediction.h5 --filelist "$LABELPATH/$TEST.csv" --filelist-header --out "$final_predictions" --out-header

echo "Done!"
echo "Predictions are in $final_predictions"