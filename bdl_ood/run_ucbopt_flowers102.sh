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
savedir=${traindir}/ood_flowers102/${ts}
mkdir -p ${savedir}

python -u run.py ${traindir} -ts ${testsamples} -tr ${testrepeat} \
  -dd ${datadir} -sd ${savedir} -d ${device} -so --ood_dataset flowers102 \
  |& tee -a ${savedir}/stdout.log