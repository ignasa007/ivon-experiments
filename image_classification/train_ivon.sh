#!/bin/bash

ts=$(date "+%Y-%m-%d-%H-%M-%S")
datadir=../datasets
optimizer=ivon

dataset=${1}  # cifar10/cifar100/tinyimagenet
model=${2}  # resnet20/resnet18wide/preresnet110/densenet121
hess_approx=${3}
seed=${4}

epochs=${epochs:-200}
device=${device:-cuda}
lr=${lr:-0.2}
lr_final=${lr_final:-0.0}
momentum=${momentum:-0.9}
momentum_hess=${momentum_hess:-0.99999}
wdecay=${wdecay:-2e-4}
tbatch=${tbatch:-50}
vbatch=${vbatch:-50}
hess_init=${hess_init:-0.5}
split=${split:-1.0}

case ${dataset} in
    cifar10 | cifar100)
        ess=50000
        ;;
    tinyimagenet)
        ess=200000
        ;;
    *)
        echo -n "unknown dataset: ${dataset}"
        exit 1
        ;;
esac

savedir=../results/${dataset}/${model}/${optimizer}-${hess_approx}/seed=${seed}/${ts}
mkdir -p ${savedir}
python -u train.py ${model} ${dataset} -opt ${optimizer} -s ${seed} -dd ${datadir} -sd ${savedir} \
    -lr ${lr} --lr_final ${lr_final} --momentum ${momentum} --momentum_hess ${momentum_hess} \
    --weight-decay ${wdecay} --hess_approx ${hess_approx} --hess_init ${hess_init} --ess ${ess} \
    --epochs ${epochs} --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} \
    --tvsplit ${split} |& tee -a ${savedir}/stdout.log