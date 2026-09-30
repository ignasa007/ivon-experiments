#!/bin/bash

time_stamp=$(date "+%Y-%m-%d-%H-%M-%S")
data_dir="${HOME}/ivon-experiments/datasets"
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

if [[ -n "${resume_dir:-}" ]]; then
    if [[ ! -d "${resume_dir}" ]]; then
        echo "Error: resume_dir = ${resume_dir} does not exist";
        exit 1
    fi
    save_dir="${resume_dir}"
else
    save_dir="${HOME}/ivon-experiments/results/${dataset}/${model}/${optimizer}/seed=${seed}/${time_stamp}"
    mkdir -p ${save_dir}
fi

python -u main.py ${model} ${dataset} -opt ${optimizer} -s ${seed} -dd ${data_dir} -sd ${save_dir} \
    ${resume_dir:+--resume} \
    -lr ${lr} --lr_final ${lr_final} --momentum ${momentum} --weight-decay ${wdecay} \
    --epochs ${epochs} --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} --tvsplit ${split} \
    |& tee -a ${save_dir}/stdout.log