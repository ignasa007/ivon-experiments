#!/bin/bash

ts=$(date "+%Y-%m-%d-%H-%M-%S")
datadir="${HOME}/ivon-experiments/datasets"
optimizer=adam

dataset=${1}
model=${2}
seed=${3}

epochs=${epochs:-200}
device=${device:-cuda}  # cpu/cuda/cuda:X
lr=${lr:-0.002}
lr_final=${lr_final:-0.0}
momentum=${momentum:-0.9}
momentum_hess=${momentum_hess:-0.999}
wdecay=${wdecay:-2e-4}
eps=${wdecay:-1e-8}
tbatch=${tbatch:-50}
vbatch=${vbatch:-50}
split=${split:-1.0}

savedir="${HOME}/ivon-experiments/results/${dataset}/${model}/${optimizer}w/seed=${seed}/${ts}"
mkdir -p ${savedir}
python -u main.py ${model} ${dataset} -opt ${optimizer} -s $seed -dd ${datadir} -sd ${savedir} \
    -lr ${lr} --lr_final ${lr_final} --momentum ${momentum} --momentum_hess ${momentum_hess} \
    --weight-decay ${wdecay} --coupled --eps ${eps} --epochs ${epochs} \
    --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} \
    --tvsplit ${split} |& tee -a ${savedir}/stdout.log