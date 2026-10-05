"""
Verify that the GGN and Hessian diagonals match.
"""


from functools import partial

import numpy as np
from torch.func import functional_call, jvp, vjp
from torchvision import datasets
from torch import optim
import matplotlib.pyplot as plt

from constants import *
from utilities import MLP, compute_hess_diag, train_gd


def compute_ggn_diag(X, Y, model, loss_type, optimizer, device, hutchinson_samples):
    def f(params):
        return functional_call(model, dict(params), (X,))
    names = [name for name, _ in model.named_parameters()]
    params = dict(model.named_parameters())
    sum_samples = 0.
    for i in range(1, hutchinson_samples+1):
        random_vectors = {name: torch.randint_like(params[name], 2).mul(2).sub(1) for name in names}
        _, Jv = jvp(f, (params,), (random_vectors,))
        _, vjp_fn = vjp(f, params)
        JtJv = vjp_fn(Jv)[0]
        sum_samples += 1/X.size(0) * torch.cat([(random_vectors[name]*JtJv[name]).reshape(-1) for name in names])
    ggn_diag = sum_samples / hutchinson_samples
    return ggn_diag


if __name__ == "__main__":

    OUTPUT_DIM = 10
    MODEL, WIDTHS, MODEL_KWARGS = MLP, [32]*1, dict()
    ACTIVATION, ACT_KWARGS = "ReLU", dict()
    LOSS_TYPE = "Mean Squared Error"
    OPTIMIZER, OPTIM_KWARGS = optim.Adam, dict(lr=1e-3, betas=(0.9, 0.95))
    EPOCHS = 1000; LOG_EVERY = EPOCHS // CKPTS

    assets, out = train_gd(
        Dataset=datasets.MNIST, output_dim=OUTPUT_DIM, subset_size=SUBSET_SIZE,
        Model=MODEL, widths=WIDTHS, model_kwargs=MODEL_KWARGS,
        act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
        loss_type=LOSS_TYPE, Optimizer=OPTIMIZER, optim_kwargs=OPTIM_KWARGS,
        epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED,
        tracker_fns=(partial(compute_ggn_diag, hutchinson_samples=5_000), partial(compute_hess_diag, hutchinson_samples=5_000))
    )

    losses, ggn_diags, hess_diags = out
    ckpts = [1, 3, 5, 7, 9, 10]
    xlabel = "GGN Diagonal"
    ylabel = "Hessian Diagonal"

    fig, axs = plt.subplots(2, 3, figsize=(7.5*3, 4.5*2))
    for i, (ckpt, ax) in enumerate(zip(ckpts, axs.flatten())):
        x, y = np.asarray(ggn_diags[ckpt]), np.asarray(hess_diags[ckpt])
        ax.scatter(x, y, s=1, color="yellowgreen", label="Data" if i==0 else None)
        identity = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), num=2)
        ax.plot(identity, identity, linewidth=5, linestyle="--", color="green", label="Identity" if i==0 else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
        ax.legend(fontsize=20, framealpha=1, markerscale=8)
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.set_title(f"Checkpoint {ckpt*LOG_EVERY}", fontsize=20)
    for ax in axs[-1,:]:
        ax.set_xlabel(xlabel, fontsize=20)
    for ax in axs[:,0]:
        ax.set_ylabel(ylabel, fontsize=20)

    fig.tight_layout()
    save_fn = f"{OPTIMIZER.__name__.lower()}/check-ggn-vs-hess.png"
    if save_fn is not None:
        plt.savefig(save_fn)
    plt.close()