#!/bin/bash

time_stamp=$(date "+%Y-%m-%d-%H-%M-%S")
data_dir="${HOME}/ivon-experiments/datasets"
optimizer=vadam

dataset=${1}
model=${2}
seed=${3}

epochs=${epochs:-200}
device=${device:-cuda}
lr=${lr:-0.002}
lr_final=${lr_final:-0.0}
beta1=${beta1:-0.9}
beta2=${beta2:-0.999}
wdecay=${wdecay:-2e-4}
tbatch=${tbatch:-50}
vbatch=${vbatch:-50}
split=${split:-1.0}

case ${dataset} in
	cifar10 | cifar100)
		ess=50000
		;;
	tinyimagenet)
		ess=200000
		;;
	*)
		echo -n "Error: unknown dataset = ${dataset}"
		exit 1
		;;
esac

opt_name="${optimizer}"
if [ -n "${coupled+x}" ]; then
    opt_name="${opt_name}-coupled"
else
    opt_name="${opt_name}-decoupled"
fi
if [ -n "${at_mean+x}" ]; then
    opt_name="${opt_name}-atmean"
fi

if [[ -n "${resume_dir:-}" ]]; then
    if [[ ! -d "${resume_dir}" ]]; then
        echo "Error: resume_dir = ${resume_dir} does not exist";
        exit 1
    fi
    save_dir="${resume_dir}"
else
    save_dir="${HOME}/ivon-experiments/results/${dataset}/${model}/${opt_name}/seed=${seed}/${time_stamp}"
    mkdir -p ${save_dir}
fi

python -u main.py ${model} ${dataset} -opt ${optimizer} -s ${seed} -dd ${data_dir} -sd ${save_dir} \
    ${resume_dir:+--resume} \
	-lr ${lr} --lr_final ${lr_final} --beta1 ${beta1} --beta2 ${beta2} \
	--weight-decay ${wdecay} ${coupled:+--coupled} --ess ${ess} ${at_mean:+--at_mean} \
	--epochs ${epochs} --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} --tvsplit ${split} \
	|& tee -a ${save_dir}/stdout.log