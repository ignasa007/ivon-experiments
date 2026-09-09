#!/bin/bash

ts=$(date "+%Y-%m-%d-%H-%M-%S")
datadir=../datasets
optimizer=sgd

dataset=$1  # cifar10/cifar100/tinyimagenet
model=$2  # resnet20/resnet18wide/preresnet110/densenet121
seed=${3:-null}

epochs=${epochs:-200}
device=${device:-cuda}
lr=${lr:-0.1}
lr_final=${lr_final:-0.0}
wdecay=${wdecay:-2e-4}
momentum=${momentum:-0.9}
tbatch=${tbatch:-50}
vbatch=${vbatch:-50}
split=${split:-1.0}

savedir=../trained/${dataset}/${model}/${optimizer}/seed=${seed}/${ts}
mkdir -p ${savedir}
python -u train.py ${model} ${dataset} -opt ${optimizer} -s $seed -dd ${datadir} -sd ${savedir} \
    -lr ${lr} --lr_final ${lr_final} --momentum ${momentum} --weight-decay ${wdecay} \
    --epochs ${epochs} --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} \
    --tvsplit ${split} |& tee -a ${savedir}/stdout.log