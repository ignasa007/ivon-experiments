#!/bin/bash
ts=$(date "+%Y-%m-%d-%H-%M-%S")
datadir=../datasets
dataset=${1}  # cifar10/cifar100/tinyimagenet
model=${2}  # resnet20/resnet18wide/preresnet110/densenet121
optimizer=ucbopt
epochs=200
lr=0.2
momentum=0.9
momentum_hess=0.99999
hess_init=${hess_init:-0.5}
wdecay=2e-4
tbatch=50
vbatch=50
split=1.0
seed=${seed:-0}
device=${device:-cuda}  # cpu/cuda/cuda:X

savedir=../final-results/${dataset}/${model}/${optimizer}/seed=${seed}/${ts}
mkdir -p ${savedir}

python -u train.py ${model} ${dataset} -opt ${optimizer} -s ${seed} -dd ${datadir} \
       -sd ${savedir} -lr ${lr} -e ${epochs} --weight-decay ${wdecay} \
       --momentum ${momentum} --momentum_hess ${momentum_hess} --hess_init ${hess_init} \
       ${decoupled_wd:+--decoupled_wd} ${bias_corr:+--bias_corr} ${rescale_lr:+--rescale_lr} \
       ${beta3:+--beta3 ${beta3}} ${gamma:+--gamma ${gamma}} \
       ${perturb_rad:+--perturb_rad ${perturb_rad}} ${clip_radius:+--clip_radius ${clip_radius}} \
       --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} \
       --tvsplit ${split} |& tee -a ${savedir}/stdout.log