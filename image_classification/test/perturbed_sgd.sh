#!/bin/bash

ts=$(date "+%Y-%m-%d-%H-%M-%S")
datadir="${HOME}/ivon-experiments/datasets"
optimizer=perturbedsgd
device=cuda
seed=0

dataset=${1}
model=${2}
testrepeat=${3:-0}

traindir="${HOME}/ivon-experiments/results/${dataset}/${model}/${optimizer}"
savedir="${traindir}/test/${ts}"

mkdir -p ${savedir}
python -u main.py ${traindir} ${dataset} -tr ${testrepeat} -s ${seed} \
    -dd ${datadir} -sd ${savedir} -d ${device} -pd -so \
    |& tee -a "${savedir}/stdout.log"