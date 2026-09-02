#!/bin/bash
ts=$(date "+%Y-%m-%d-%H-%M-%S")
datadir=../datasets
dataset=${1}  # cifar10/cifar100/tinyimagenet
model=${2}  # resnet20/resnet18wide/preresnet110/densenet121
optimizer=ivadam
epochs=200
device=cuda  # cpu/cuda/cuda:X
lr=0.002
momentum=0.9
momentum_hess=0.95
wdecay=2e-4
tbatch=50
vbatch=50
split=1.0
seed=${3:-null}

case $dataset in

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

if [ -n "${coupled_wd+x}" ]; then
    opt_name="${optimizer}-coupled"
else
    opt_name="${optimizer}-decoupled"
fi
savedir=../results/${dataset}/${model}/${opt_name}/seed=${seed}/${ts}

mkdir -p ${savedir}
python -u train.py ${model} ${dataset} -opt ${optimizer} -s ${seed} -dd ${datadir} \
       -sd ${savedir} -lr ${lr} -e ${epochs} --weight-decay ${wdecay} ${coupled_wd:+--coupled_wd} \
       --momentum ${momentum} --momentum_hess ${momentum_hess} \
       --ess ${ess} --device ${device} -pd --tbatch ${tbatch} --vbatch ${vbatch} \
       --tvsplit ${split} |& tee -a ${savedir}/stdout.log