#!/bin/bash

time_stamp=$(date "+%Y-%m-%d-%H-%M-%S")
data_dir="${HOME}/ivon-experiments/datasets"
optimizer=perturbedsgd
device=cuda

dataset=${1}
model=${2}
testrepeat=${3}
ood_dataset=${4}

train_dir="${HOME}/ivon-experiments/results/${dataset}/${model}/${optimizer}"
save_dir="${train_dir}/ood/${time_stamp}"

mkdir -p ${save_dir}
python -u main.py ${train_dir} ${dataset} -tr ${testrepeat} \
    -dd ${data_dir} -sd ${save_dir} -d ${device} -so --ood_dataset ${ood_dataset} \
    |& tee -a ${save_dir}/stdout.log