#!/bin/bash
ts=$(date "+%Y-%m-%d-%H-%M-%S")
datadir=../datasets
dataset=${1}  # cifar10/cifar100/tinyimagenet
model=${2}  # resnet20/resnet18wide/preresnet110/densenet121
optimizer=ucbopt
testsamples=1
testrepeat=1
device=cuda  # cpu/cuda/cuda:X

traindir=../final-results/${dataset}/${model}/${optimizer}
savedir=${traindir}/indomain/${ts}
mkdir -p ${savedir}

python -u test_ucbopt.py ${traindir} -ts ${testsamples} -tr ${testrepeat} \
    -dd ${datadir} -sd ${savedir} -d ${device} -pd -so \
    |& tee -a ${savedir}/stdout.log