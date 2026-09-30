#!/bin/bash

time_stamp=$(date "+%Y-%m-%d-%H-%M-%S")
data_dir="${HOME}/ivon-experiments/datasets"
optimizer=sfrmsprop

dataset=${1}
model=${2}
seed=${3}

epochs=${epochs:-200}
device=${device:-cuda}
lr=${lr:-0.005}
lr_final=${lr_final:-0.0}
momentum=${momentum:-0.9}
momentum_hess=${momentum_hess:-0.999}
wdecay=${wdecay:-3e-4}
eps=${wdecay:-5e-5}
tbatch=${tbatch:-50}
vbatch=${vbatch:-50}
split=${split:-1.0}

save_dir="${HOME}/ivon-experiments/results/${dataset}/${model}/${optimizer}/seed=${seed}/${time_stamp}"
mkdir -p ${save_dir}
python -u main.py ${model} ${dataset} -opt ${optimizer} -s $seed -dd ${data_dir} -sd ${save_dir} \
    -lr ${lr} --lr_final ${lr_final} --momentum ${momentum} --momentum_hess ${momentum_hess} \
    --weight-decay ${wdecay} --eps ${eps} --epochs ${epochs} \
    --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} \
    --tvsplit ${split} |& tee -a ${save_dir}/stdout.log