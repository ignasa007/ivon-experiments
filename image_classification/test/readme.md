# Test Models on Image Classification Tasks

**Datasets.** You can choose the dataset as one of `cifar10`, `cifar100`, or `tinyimagenet`. For example, set `dataset=cifar10`.

**Models.** You can choose the model as one of `resnet20` (272K parameters), `densenet121` (1M), `preresnet110` (4M), or `resnet18wide` (11M). For example, set `model=resnet20`.

**Optimizers.** AdamW and SGD can be tested using the command `bash ${optimizer}.sh ${dataset} ${model}`. PerturbedSGD needs an extra argument &ndash; the number of MC samples used for posterior approximation; launch a run with `bash perturbed_sgd.sh ${dataset} ${model} ${mc_samples}`. IVON needs yet another argument &ndash; the Hessian-approximation strategy; launch a run with `bash ivon.sh ${dataset} ${model} ${hess_approx} ${mc_samples}`.

**Output.** The results are saved in `${traindir}/test/${timestamp}`. The outputs of a run are:

1. Print statements over training under `stdout.log`.
2. Predicitions, calibration plots, and evaluation metrics for all experiments.

**Note:** For seeds with multiple experiments, you will be asked to confirm which ones you want to include in the test. Currently, to keep the logic straight-forward, we only allow for one experiment per seed.