#!/bin/bash
ts=$(date "+%Y-%m-%d-%H-%M-%S")
datadir=../datasets
dataset=${1}  # cifar10/cifar100/tinyimagenet
model=${2}  # resnet20/resnet18wide/preresnet110/densenet121
optimizer=perturbedsgd
testrepeat=${3:-0}  # mc samples
device=cuda
traindir="../results/${dataset}/${model}/${optimizer}"
savedir="${traindir}/evaluation/${ts}"
seed=0

mkdir -p ${savedir}
python -u test.py ${traindir} ${dataset} -tr ${testrepeat} -s ${seed} \
    -dd ${datadir} -sd ${savedir} -d ${device} -pd -so |& tee -a "${savedir}/stdout.log"