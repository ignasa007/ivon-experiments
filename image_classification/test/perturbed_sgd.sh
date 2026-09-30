#!/bin/bash

time_stamp=$(date "+%Y-%m-%d-%H-%M-%S")
data_dir="${HOME}/ivon-experiments/datasets"
optimizer=perturbedsgd
device=cuda
seed=0

dataset=${1}
model=${2}
testrepeat=${3:-0}

train_dir="${HOME}/ivon-experiments/results/${dataset}/${model}/${optimizer}"
save_dir="${train_dir}/test/${time_stamp}"

mkdir -p ${save_dir}
python -u main.py ${train_dir} ${dataset} -tr ${testrepeat} -s ${seed} \
    -dd ${data_dir} -sd ${save_dir} -d ${device} -pd -so \
    |& tee -a "${save_dir}/stdout.log"