from functools import partial

from torchvision import datasets
from torch import optim

from constants import *
from utilities import MLP, compute_exp_avg_sq, compute_hess_diag, train_gd, plot


OUTPUT_DIM = 10
MODEL, WIDTHS, MODEL_KWARGS = MLP, [64]*3, dict()
ACTIVATION, ACT_KWARGS = "ReLU", dict()
LOSS_TYPE = "Mean Squared Error"
OPTIMIZER, OPTIM_KWARGS = optim.Adam, dict(lr=1e-3*100, betas=(0.0, 0.95), eps=100)
EPOCHS = 10000; LOG_EVERY = EPOCHS // CKPTS

assets, out = train_gd(
    Dataset=datasets.MNIST, output_dim=OUTPUT_DIM, subset_size=SUBSET_SIZE,
    Model=MODEL, widths=WIDTHS, model_kwargs=MODEL_KWARGS,
    act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
    loss_type=LOSS_TYPE, Optimizer=OPTIMIZER, optim_kwargs=OPTIM_KWARGS,
    epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED,
    tracker_fns=(compute_exp_avg_sq, partial(compute_hess_diag, hutchinson_samples=HUTCHINSON_SAMPLES))
)

losses, exp_avg_sqs, hess_diags = out
ckpts = [1, 3, 5, 7, 9, 10]
xlabel = "Exp Avg Sq"
ylabel = "Hessian Diagonal"

plot(exp_avg_sqs, hess_diags, ckpts, LOG_EVERY, xlabel, ylabel, save_fn=f"{OPTIMIZER.__name__.lower()}/isolating-ema.png")