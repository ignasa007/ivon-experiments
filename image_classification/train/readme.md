# Train Models on Image Classification Tasks

**Datasets.** You can choose the dataset as one of `cifar10`, `cifar100`, or `tinyimagenet`. For example, set `dataset=cifar10`.

**Models.** You can choose the model as one of `resnet20` (272K parameters), `densenet121` (1M), `preresnet110` (4M), or `resnet18wide` (11M). For example, set `model=resnet20`.

**Optimizers.** All optimizers, except IVON, can be trained using the command `bash ${optimizer}.sh ${dataset} ${model} ${seed}`. Currently, we support standard algorithms &ndash; SGD, Adam and AdamW &ndash; and some new choices, details of which can be viewed in `optimizers/readme.md`. You may toggle the arguments passed to the training script from the command line, e.g. `beta1=0.95 bash adam.sh ${dataset} ${model} ${seed}`; check the bash files for knobs available to play with.

IVON needs an extra argument &ndash; the Hessian-approximation strategy, which can be one of `hess_approx=price` and `gradsq`, with the former being the standard choice. Launch an IVON run with `bash ${optimizer}.sh ${dataset} ${model} ${hess_approx} ${seed}`. The results are saved in `results/${dataset}/${model}/${optimizer}-${hess_approx}/seed=${seed}/${timestamp}`.

**Output.** The outputs of a run are:

1. Print statements over training under `stdout.log`.
2. Checkpoints at epochs 40, 50, 75, 100, 150, 200.
3. Training and evaluation metrics, along with runtime for each epoch.

**Seed.** We choose `seed` to take five values, from 0 to 4.

**Note.** In case training fails with a checkpoint saved, e.g. failed at epoch 110 with checkpoint saved at epoch 100, then you may resume training by setting the variable `resume_dir` before running the bash scripts.