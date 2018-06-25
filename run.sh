#!/bin/bash

# Training and prediction for the QMUL Bird audio detection challenge 2017
# http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/
# Thomas Grill <thomas.grill@ofai.at>

here="${0%/*}"

# import general configuration
. "$here/config.inc"

# import spectral parametrization
. "$here/spectral_features.inc"

# import network/learning configuration
. "$here/network_${NETWORK}.inc"

LISTPATH="$WORKPATH/filelists"
SPECTPATH="$WORKPATH/spect"


# locations of prediction files
first_predictions="$WORKPATH/prediction_first.csv"
second_predictions="$WORKPATH/prediction_second.csv"
final_predictions="$WORKPATH/prediction_final.csv"

# pre-expand the list of test filelists, needed by some of the python methods
testfilelists="$LABELPATH`echo "$TEST" | sed -E "s# +#.csv,$LABELPATH#g"`.csv"

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
    echo_status "Done with training model ${model}. `date`. Final loss = ${loss}."
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
    --var filelist:lists="$filelists" \
    --var filelist:sep=',' \
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

    "$here/code/create_filelists.py" "$LABELPATH" ${TRAIN} --mode "train" --out "$LISTPATH/%(fold)s_%(num)i" || return $?
    if [ "$TEST" = "" ]; then
        echo_status "NOT computing file lists for test sets since no test sets were specified yet."
    else
        "$here/code/create_filelists.py" "$LABELPATH" ${TEST}  --mode "test"  --out "$LISTPATH/%(fold)s"         || return $?
    fi

    echo_status "Computing spectrograms."
    mkdir $SPECTPATH 2> /dev/null
    "$here/code/prepare_spectrograms.sh" "${AUDIOPATH}" "${SPECTPATH}" ${SPEC_SR} ${SPEC_FPS} ${SPEC_FFTLEN} ${SPEC_FMIN} ${SPEC_FMAX} ${SPEC_BANDS}

    echo_status "Done computing spectrograms."

    email_status "Done with stage1 preparations" "Computed filelists and spectrograms."
}

#############################
# first stage training
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
# first stage prediction
#############################
function stage1_predict {

    if [ "$TEST" = "" ]; then
        echo_status "NOT computing first stage predictions since no test sets were specified yet."
        return 1
    fi

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
    "$here/code/predict.py" "$WORKPATH"/model_first_?.prediction.h5 --filelist "$testfilelists" --filelist-header --out "$first_predictions" --out-header || return $?
    echo_status "Done. `date`. First stage predictions are in ${first_predictions}."

    email_status "Done with stage1 predictions" "First stage predictions are in ${first_predictions}."
}

#############################
# first stage validation
#############################
function stage1_validate {
    echo_status "Computing first stage validations."

    cmdargs="${@:1}"
    for i in `seq ${model_count}`; do
        model="$WORKPATH/model_first_${i}"
        validation="${model}.validation"
        if [ ! -f "${validation}.h5" ]; then # check for existence
            evaluate_model "${model}" "val_${i}" "${validation}" ${cmdargs} #|| return $?
        else
            echo_status "Using existing validations ${validation}."
        fi
    done

    first_validations="$WORKPATH/validation_first.csv"

    # prediction by bagging
    echo_status "Bagging first stage validations."
    vallists=`echo $LISTPATH/val_?`
    "$here/code/predict.py" "$WORKPATH"/model_first_?.validation.h5 --filelist ${vallists// /,} --out "$first_validations" --keep-prefix --keep-suffix --out-header --skip-missing || return $?
    filelists=""
    for t in ${TRAIN}; do
        filelists+=" ${LABELPATH}/${t}.csv"
    done
    auc=`"$here/code/evaluate_auc.py" "${first_validations}" ${filelists} --splits ${vallists// /,} --gt-header --pred-header --gt-suffix='.wav'`
    echo_status "Done. `date`. First stage validation AUC score is ${auc}."

    email_status "Done with stage1 validations" "First stage validation AUC score is ${auc}."
}

#############################
# compute pseudo_labels
#############################
function stage2_prepare {
    echo_status "Prepare second stage by analyzing first stage."

    # filter list by threshold
    # split into folds randomly
    "$here/code/make_pseudo.py" --filelist "$first_predictions" --filelist-header --threshold=${pseudo_threshold} --folds=${pseudo_folds} --out "$LISTPATH/testdata.pseudo_%(fold)i" --out-prefix-filelists="$testfilelists" --out-suffix='.wav' || return $?

    # merge each train filelist with a pseudo filelist
    for i in `seq ${model_count}`; do
        for h in `seq ${pseudo_folds}`; do
            cat "$LISTPATH/train_${i}" "$LISTPATH/testdata.pseudo_${h}" > "$LISTPATH/train_${i}_pseudo_${h}"
        done
    done
    echo_status "Prepared file lists for second stage."

    email_status "Done with stage2 preparations" "Generated pseudo-labeled training data."
}

#############################
# second stage training
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
# second stage prediction
# by bagging all available models
############################################
function stage2_predict {

    if [ "$TEST" = "" ]; then
        echo_status "NOT computing final predictions since no test sets were specified yet."
        return 1
    fi

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
    "$here/code/predict.py" "$WORKPATH"/model_second*.prediction.h5 --filelist "$testfilelists" --filelist-header --out "$second_predictions" --out-header || return $?
    "$here/code/predict.py" "$WORKPATH"/model_*.prediction.h5 --filelist "$testfilelists" --filelist-header --out "$final_predictions" --out-header || return $?
    echo_status "Done. `date`. Final predictions are in ${final_predictions}."

    email_status "Done with stage2 predictions" "Final predictions are in ${final_predictions}."
}

#############################
# second stage validation
#############################
function stage2_validate {
    echo_status "Computing second stage validations."

    cmdargs="${@:1}"
    for i in `seq ${model_count}`; do
        for h in `seq ${pseudo_folds}`; do
            model="$WORKPATH/model_second_${i}_${h}"
            validation="${model}.validation"
            if [ ! -f "${validation}.h5" ]; then # check for existence
                evaluate_model "${model}" "val_${i}" "${validation}" ${cmdargs} #|| return $?
            else
                echo_status "Using existing validations ${validation}."
            fi
        done
    done

    second_validations="$WORKPATH/validation_second.csv"

    # prediction by bagging
    echo_status "Bagging first stage validations."
    vallists=`echo $LISTPATH/val_?`
    "$here/code/predict.py" "$WORKPATH"/model_second_?_?.validation.h5 --filelist ${vallists// /,} --out "$second_validations" --keep-prefix --keep-suffix --out-header --skip-missing || return $?
    filelists=""
    for t in ${TRAIN}; do
        filelists+=" ${LABELPATH}/${t}.csv"
    done
    auc=`"$here/code/evaluate_auc.py" "${second_validations}" ${filelists} --splits ${vallists// /,} --gt-header --pred-header --gt-suffix='.wav'`
    echo_status "Done. `date`. Second stage validation AUC score is ${auc}."

    email_status "Done with stage2 validations" "Second stage validation AUC score is ${auc}."
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
    stage1_prepare ${cmdargs} && stage1_train ${cmdargs} && stage1_validate ${cmdargs} && stage1_predict ${cmdargs}
    ### Stage 2 deactivated for DCASE 2018 baseline - it might work, but hasn't been fully tested
    # && stage2_prepare ${cmdargs} # && stage2_train ${cmdargs} && stage2_validate ${cmdargs} && stage2_predict ${cmdargs}
else
    echo_info "Running sub-task ${1}:"
    ${@:1}
fi
