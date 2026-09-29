#!/bin/bash
ts=$(date "+%Y-%m-%d-%H-%M-%S")
datadir=../datasets
dataset=${1}  # cifar10/cifar100/tinyimagenet
model=${2}  # resnet20/resnet18wide/preresnet110/densenet121
hess_approx=${3}
optimizer=ivon
testrepeat=${4:-0}  # mc samples
device=cuda
traindir="../results/${dataset}/${model}/${optimizer}-${hess_approx}"
savedir="${traindir}/evaluation/${ts}"
seed=0

mkdir -p ${savedir}
python -u test.py ${traindir} ${dataset} -tr ${testrepeat} -s ${seed} \
    -dd ${datadir} -sd ${savedir} -d ${device} -pd -so |& tee -a ${savedir}/stdout.log