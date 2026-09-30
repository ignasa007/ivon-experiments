#!/bin/bash

ts=$(date "+%Y-%m-%d-%H-%M-%S")
datadir="${HOME}/ivon-experiments/datasets"
optimizer=adamw
device=cuda
seed=0

dataset=${1}
model=${2}

traindir="${HOME}/ivon-experiments/results/${dataset}/${model}/${optimizer}-${hess_approx}"
savedir="${traindir}/test/${ts}"

mkdir -p ${savedir}
python -u main.py ${traindir} ${dataset} -s ${seed} \
    -dd ${datadir} -sd ${savedir} -d ${device} -pd -so \
    |& tee -a ${traindir}/stdout-${ts}.log