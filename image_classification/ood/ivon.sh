#!/bin/bash

time_stamp=$(date "+%Y-%m-%d-%H-%M-%S")
data_dir="${HOME}/ivon-experiments/datasets"
optimizer=ivon
device=cuda

dataset=${1}
model=${2}
hess_approx=${3}
testrepeat=${4}
ood_dataset=${5}

train_dir="${HOME}/ivon-experiments/results/${dataset}/${model}/${optimizer}-${hess_approx}"
save_dir="${train_dir}/ood/${time_stamp}"

mkdir -p ${save_dir}
python -u main.py ${train_dir} ${dataset} -tr ${testrepeat} \
    -dd ${data_dir} -sd ${save_dir} -d ${device} -so --ood_dataset ${ood_dataset} \
    |& tee -a ${save_dir}/stdout.log