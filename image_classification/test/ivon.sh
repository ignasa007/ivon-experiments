#!/bin/bash

time_stamp=$(date "+%Y-%m-%d-%H-%M-%S")
data_dir="${HOME}/ivon-experiments/datasets"
optimizer=ivon
device=cuda
seed=0

dataset=${1}
model=${2}
hess_approx=${3}
test_repeat=${4:-0}

train_dir="${HOME}/ivon-experiments/results/${dataset}/${model}/${optimizer}-${hess_approx}"
save_dir="${train_dir}/test/${time_stamp}"

mkdir -p ${save_dir}
python -u main.py ${train_dir} ${dataset} -tr ${test_repeat} -s ${seed} \
    -dd ${data_dir} -sd ${save_dir} -d ${device} -pd -so \
    |& tee -a ${save_dir}/stdout.log