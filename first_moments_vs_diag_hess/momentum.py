from functools import partial

import torchvision
import torch.optim as optim

from constants import *
from utilities import MLP, compute_momentum, compute_hess_diag, train_gd, plot


OUTPUT_DIM = 10
MODEL, WIDTHS, MODEL_KWARGS = MLP, [64]*3, dict()
ACTIVATION, ACT_KWARGS = "ReLU", dict()
LOSS_TYPE = "Mean Squared Error"
OPTIMIZER, OPTIM_KWARGS = optim.SGD, dict(lr=1e-3, momentum=0.9)
EPOCHS = 10000; LOG_EVERY = EPOCHS // CKPTS

assets, out = train_gd(
    Dataset=torchvision.datasets.MNIST, output_dim=OUTPUT_DIM, subset_size=SUBSET_SIZE,
    Model=MODEL, widths=WIDTHS, model_kwargs=MODEL_KWARGS,
    act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
    loss_type=LOSS_TYPE, Optimizer=OPTIMIZER, optim_kwargs=OPTIM_KWARGS,
    epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED,
    tracker_fns=(compute_momentum, partial(compute_hess_diag, hutchinson_samples=HUTCHINSON_SAMPLES))
)

losses, grads, hess_diags = out
ckpts = [1, 3, 5, 7, 9, 10]
xlabel = "Momentum"
ylabel = "Hessian Diagonal"

plot(grads, hess_diags, ckpts, LOG_EVERY, xlabel, ylabel, save_fn=f"{ASSETS}/momentum.png")