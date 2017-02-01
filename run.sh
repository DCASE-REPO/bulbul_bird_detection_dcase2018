#!/bin/bash

# Training and prediction for the QMUL Bird audio detection challenge 2017
# http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/
# Thomas Grill <thomas.grill@ofai.at>

here="${0%/*}"

. "$here/config.inc"

# import network/learning configuration
. "$here/network_${NETWORK}.inc"

LISTPATH="$WORKPATH/filelists"
SPECTPATH="$WORKPATH/spect"


# locations of prediction files
first_predictions="$WORKPATH/prediction_first.csv"
second_predictions="$WORKPATH/prediction_second.csv"
final_predictions="$WORKPATH/prediction_final.csv"


# colored text if possible
if command -v tput >/dev/null 2>&1; then
    text_bold=$(tput bold)
    text_boldblue=$(tput setaf 4)
    text_normal=$(tput sgr0)
else
    text_bold=
    text_boldblue=
    text_normal=
fi

function echo_info {
    echo -e "${text_boldblue}${@}${text_normal}"
}
function echo_status {
    echo -e "${text_bold}${@}${text_normal}"
}

function email_status {
    if [ "${EMAIL}" != "" ]; then
        echo -e "Subject: run.sh - ${1}\n${@:2}" | sendmail "${EMAIL}"
    fi
}

#############################
# define training
#############################
function train_model {
    model="$1"  # model including path
    filelists="$2"  # file list to use
    extralabels="$3"
    seed="$4"
    cmdargs="${@:5}"

    echo_status "Computing model ${model} with network ${NETWORK}."

    "$here/code/simplenn_main.py" \
    --mode=train \
    --problem=binary \
    --var measures= \
    --inputs filelist:filelist \
    --var filelist:path="$LISTPATH" \
    --var filelist:lists="${filelists}" \
    --var filelist:sep=',' \
    --var filelist:column=0 \
    --process "filelistshuffle:shuffle(seed=$seed,memory=25000)" \
    --process "input:${here}/code/load_data.py(type=spect,downmix=0,cycle=0,denoise=1,width=${net_width},seed=$seed)" \
    --var input:labels="${LABELPATH}"/'*.csv',"${extralabels}" \
    --var input:data="${SPECTPATH}/%(id)s.h5" \
    --var input:data_vars=1k \
    --process collect:collect \
    --var "collect:source=0..1"  \
    --process "scale@1:range(out_min=0.01,out_max=0.99)" \
    --layers "${net_layers}" \
    --save "${model}.h5" \
    ${net_options} \
    ${cmdargs} || return $?

    loss=`python -c  'import h5py,sys; print h5py.File(sys.argv[1]+".h5","r")["training"]["train_loss_epoch"][-1]' ${model}`
    echo_status "Done with training model ${model}. Final loss = ${loss}."
    email_status "Done with training model ${model}" "Final loss = ${loss}."
}

#############################
# define evaluation
#############################
function evaluate_model {
    model="$1"  # model including path
    filelists="$2"  # file list to use
    predictions="$3"  # model including path
    cmdargs="${@:4}" # extra arguments

    echo_status "Evaluating model ${model}."

    "$here/code/simplenn_main.py" \
    --mode=evaluate \
    --var input:labels="${LABELPATH}"/'*.csv' \
    --var input:data="${SPECTPATH}/%(id)s.h5" \
    --var input:targets_needed=0 \
    --var filelist:path="$LISTPATH" \
    --var filelist:lists=$filelists \
    --var filelistshuffle:bypass=1 \
    --var augment:bypass=1 \
    --load "${model}.h5" \
    --save "${predictions}.h5" \
    ${cmdargs}
}

#####################################
# prepare file lists and spectrograms
#####################################
function stage1_prepare {
    echo_status "Preparing file lists."
    mkdir $LISTPATH 2> /dev/null

    "$here/code/create_filelists.py" "$LABELPATH" ${TRAIN} --out "$LISTPATH/%(fold)s_%(num)i" --num ${model_count} --folds "train=$((model_count-1)),val=1" || return $?
    "$here/code/create_filelists.py" "$LABELPATH" ${TEST} --out "$LISTPATH/%(fold)s" --num ${model_count} --folds "test=1"  || return $?

    echo_status "Computing spectrograms."
    mkdir $SPECTPATH 2> /dev/null
    "$here/code/prepare_spectrograms.sh" "${AUDIOPATH}" "${SPECTPATH}"
    echo_status "Done computing spectrograms."

    email_status "Done with stage1 preparations" "Computed filelists and spectrograms."
}

#############################
# first training run
#############################
function stage1_train {
    echo_status "First training stage."

    # process model and fold indices
    if [ "$1" != "" -a  "${1:0:1}" != '-' ]; then
        # index is given as first argument
        idxs="$1"
        cmdargs="${@:2}"
    else
        idxs=`seq ${model_count}`
        cmdargs="${@:1}"
    fi

    for i in ${idxs}; do
        model="$WORKPATH/model_first_${i}"
        if [ ! -f "${model}.h5" ]; then # check for existence
            echo_status "Training model ${model}."
            train_model "${model}" "train_${i}" '' ${i} ${cmdargs} || return $?
            echo_status "Done training model ${model}."
        else
            echo_status "Using existing model ${model}."
        fi
    done
}

#############################
# first prediction run
#############################
function stage1_predict {
    echo_status "Computing first stage predictions."

    cmdargs="${@:1}"
    for i in `seq ${model_count}`; do
        model="$WORKPATH/model_first_${i}"
        prediction="${model}.prediction"
        if [ ! -f "${prediction}.h5" ]; then # check for existence
            evaluate_model "${model}" "test" "${prediction}" ${cmdargs} || return $?
        else
            echo_status "Using existing predictions ${prediction}."
        fi
    done

    # prediction by bagging
    echo_status "Bagging first stage predictions."
    "$here/code/predict.py" "$WORKPATH"/model_first_?.prediction.h5 --filelist "$LABELPATH/$TEST.csv" --filelist-header --out "$first_predictions" --out-header || return $?
    echo_status "Done. First stage predictions are in ${first_predictions}."

    email_status "Done with stage1 predictions" "First stage predictions are in ${first_predictions}."
}

#############################
# compute pseudo_labels
#############################
function stage2_prepare {
    echo_status "Prepare second stage by analyzing first stage."

    # filter list by threshold
    # split in half randomly
    "$here/code/make_pseudo.py" --filelist "$first_predictions" --filelist-header --threshold=${pseudo_threshold} --folds=${pseudo_folds} --out "$LISTPATH/testdata.pseudo_%(fold)i" --out-prefix="$TEST/" --out-suffix='.wav' || return $?

    # merge train filelist and half pseudo filelists
    for i in `seq ${model_count}`; do
        for h in `seq ${pseudo_folds}`; do
            cat "$LISTPATH/train_${i}" "$LISTPATH/testdata.pseudo_${h}" > "$LISTPATH/train_${i}_pseudo_${h}"
        done
    done
    echo_status "Prepared file lists for second stage."

    email_status "Done with stage2 preparations" "Generated pseudo-labeled training data."
}

#############################
# second run
#############################
function stage2_train {
    echo_status "Second training stage."

    # process model and fold indices
    if [ "$1" != "" -a "${1:0:1}" != '-' ]; then
        # index is given as first argument
        idxs="$1"
        if [ "$2" != "" -a "${2:0:1}" != '-' ]; then
            # index is given as second argument
            folds="$2"
            cmdargs="${@:3}"
        else
            folds=`seq ${pseudo_folds}`
            cmdargs="${@:2}"
        fi
    else
        idxs=`seq ${model_count}`
        folds=`seq ${pseudo_folds}`
        cmdargs="${@:1}"
    fi

    for i in $idxs; do
        for h in $folds; do
            model="$WORKPATH/model_second_${i}_${h}"
            if [ ! -f "${model}.h5" ]; then # check for existence
                echo_status "Training model ${model}."
                train_model "${model}" "train_${i}_pseudo_${h}" "$LISTPATH/testdata.pseudo_*" ${i} ${cmdargs} || return $?
                echo_status "Done training model ${model}."
            else
                echo_status "Using existing model ${model}."
            fi
        done
    done
}

#############################################
# prediction by bagging all available models
############################################
function stage2_predict {
    echo_status "Computing final predictions."

    cmdargs="${@:1}"
    for i in `seq ${model_count}`; do
        for h in `seq ${pseudo_folds}`; do
            model="$WORKPATH/model_second_${i}_${h}"
            prediction="${model}.prediction"
            if [ ! -f "${prediction}.h5" ]; then # check for existence
                evaluate_model "${model}" test "${prediction}" ${cmdargs} || return $?
            else
                echo_status "Using existing predictions ${prediction}."
            fi
        done
    done

    echo_status "Bagging final predictions."
    "$here/code/predict.py" "$WORKPATH"/model_second*.prediction.h5 --filelist "$LABELPATH/$TEST.csv" --filelist-header --out "$second_predictions" --out-header || return $?
    "$here/code/predict.py" "$WORKPATH"/model_*.prediction.h5 --filelist "$LABELPATH/$TEST.csv" --filelist-header --out "$final_predictions" --out-header || return $?
    echo_status "Done. Final predictions are in ${final_predictions}."

    email_status "Done with stage2 predictions" "Final predictions are in ${final_predictions}."
}

###################################################################

if [ "$1" == 'help' -o "$1" == '-help' -o "$1" == '--help' ]; then
    echo_info "Proposal for the Bird audio detection challenge 2017"
    echo_info "See http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge"
    echo_info "by Thomas Grill <thomas.grill@ofai.at>"
    echo_info ""
    echo_info "Without any arguments, the full two-stage train/predict sequence is run"
    echo_info "Subtasks can be run by specifying one of: stage1_prepare, stage1_train, stage1_predict, stage2_prepare, stage2_train, stage2_predict"
elif [ "$1" == "" -o "${1:0:1}" == '-' ]; then
    echo_info "Running full two-stage train/predict sequence:"
    cmdargs="${@:1}"
    stage1_prepare ${cmdargs} && stage1_train ${cmdargs} && stage1_predict ${cmdargs} && stage2_prepare ${cmdargs} && stage2_train ${cmdargs} && stage2_predict ${cmdargs} 
else
    echo_info "Running sub-task ${1}:"
    ${@:1}
fi
