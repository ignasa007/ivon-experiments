#!/bin/bash

ts=$(date "+%Y-%m-%d-%H-%M-%S")
datadir="${HOME}/ivon-experiments/datasets"
optimizer=ivon
device=cuda

dataset=${1}
model=${2}
hess_approx=${3}
testrepeat=${4}
ood_dataset=${5}

traindir="${HOME}/ivon-experiments/results/${dataset}/${model}/${optimizer}-${hess_approx}"
savedir="${traindir}/ood/${ts}"

mkdir -p ${savedir}
python -u main.py ${traindir} ${dataset} -tr ${testrepeat} \
    -dd ${datadir} -sd ${savedir} -d ${device} -so --ood_dataset ${ood_dataset} \
    |& tee -a ${savedir}/stdout.log