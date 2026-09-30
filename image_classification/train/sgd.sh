#!/bin/bash

ts=$(date "+%Y-%m-%d-%H-%M-%S")
datadir="${HOME}/ivon-experiments/datasets"
optimizer=sgd

dataset=${1}
model=${2}
seed=${3}

epochs=${epochs:-200}
device=${device:-cuda}
lr=${lr:-0.1}
lr_final=${lr_final:-0.0}
wdecay=${wdecay:-2e-4}
momentum=${momentum:-0.9}
tbatch=${tbatch:-50}
vbatch=${vbatch:-50}
split=${split:-1.0}

savedir="${HOME}/ivon-experiments/trained/${dataset}/${model}/${optimizer}/seed=${seed}/${ts}"
mkdir -p ${savedir}
python -u train.py ${model} ${dataset} -opt ${optimizer} -s $seed -dd ${datadir} -sd ${savedir} \
    -lr ${lr} --lr_final ${lr_final} --momentum ${momentum} --weight-decay ${wdecay} \
    --epochs ${epochs} --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} \
    --tvsplit ${split} |& tee -a ${savedir}/stdout.log