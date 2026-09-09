#!/bin/bash

ts=$(date "+%Y-%m-%d-%H-%M-%S")
datadir=../datasets
optimizer=ivadam

dataset=${1}  # cifar10/cifar100/tinyimagenet
model=${2}  # resnet20/resnet18wide/preresnet110/densenet121
seed=${3}

epochs=${epochs:-200}
device=${device:-cuda}
lr=${lr:-0.002}
lr_final=${lr:-0.0}
momentum=${momentum:-0.9}
momentum_hess=${momentum_hess:-0.95}
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
		echo -n "unknown dataset: ${dataset}"
		exit 1
		;;
esac

opt_name="${optimizer}"
if [ -n "${coupled_wd+x}" ]; then
    opt_name="${opt_name}-coupled"
else
    opt_name="${opt_name}-decoupled"
fi
if [ -n "${at_mean+x}" ]; then
    opt_name="${opt_name}-atmean"
fi
savedir=../results/${dataset}/${model}/${opt_name}/seed=${seed}/${ts}
mkdir -p ${savedir}
python -u train.py ${model} ${dataset} -opt ${optimizer} -s ${seed} -dd ${datadir} -sd ${savedir} \
	-lr ${lr} --lr_final ${lr_final} --momentum ${momentum} --momentum_hess ${momentum_hess} \
	--weight-decay ${wdecay} ${coupled_wd:+--coupled_wd} --ess ${ess} ${at_mean:+--at_mean} \
	--epochs ${epochs} --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} \
	--tvsplit ${split} |& tee -a ${savedir}/stdout.log