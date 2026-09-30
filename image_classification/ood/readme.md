# Test Models on Image Classification Tasks

**Datasets.** You can choose the dataset as one of `cifar10`, `cifar100`, or `tinyimagenet`. For example, set `dataset=cifar10`. The OOD dataset can be one of `svhn` and `flowers102`. For example, set `ood_dataset=svhn`.

**Models.** You can choose the model as one of `resnet20` (272K parameters), `densenet121` (1M), `preresnet110` (4M), or `resnet18wide` (11M). For example, set `model=resnet20`.

**Optimizers.** PerturbedSGD can be tested using the command `bash perturbed_sgd.sh ${dataset} ${model} ${mc_samples} ${ood_dataset}`. IVON needs an extra argument &ndash; the Hessian-approximation strategy; launch a run with `bash ivon.sh ${dataset} ${model} ${hess_approx} ${mc_samples} ${ood_dataset}`.

**Output.** The results are saved in `${traindir}/ood/${timestamp}`. The outputs of a run are:

1. Print statements over training under `stdout.log`.
2. Predicitions on in-domain and OOD datasets for all experiments.

**Note:** For seeds with multiple experiments, you will be asked to confirm which ones you want to include in the test. Currently, to keep the logic straight-forward, we only allow for one experiment per seed.