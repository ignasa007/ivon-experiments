'''
python -m hutchinson.plot_ivon
'''


import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt

from common.trainutils import loadcheckpoint


def plot(ax, x, y, checkpoint, label_data=True):
    mask = (x > 1e-24) & (y > 1e-24)
    x, y = x[mask], y[mask]
    ax.scatter(x, y, s=1, color="cornflowerblue", label="Data" if label_data else None)
    ax.vlines(
        # h_0 = 0.5, beta_2 = 1-1e-5, batch_size = 50, iters per epoch = 50k/50 = 1k
        0.5*(1-1e-5)**(int(checkpoint)*1000), y.min(), y.max(), linewidth=5, linestyle="--",
        color="black", label=r"$\mathbf{h}_0 \cdot \beta_2^t$" if label_data else None
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    if label_data:
        ax.legend(fontsize=20, framealpha=1., markerscale=8)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_title(f"Checkpoint {checkpoint}", fontsize=20)

def ivon_hess(optimizer):
    out = torch.cat([
        group.get("hess", torch.zeros(group["numel"]))
        for group in optimizer.param_groups
    ])
    return out

def main(exp_dir, approx_func, data_samples, hutchinson_samples, nrows, ncols):

    save_dir = f"{exp_dir}/hutchinson/ds={data_samples}_hs={hutchinson_samples}"

    fig, axs = plt.subplots(nrows, ncols, figsize=(7.5*ncols, 4.5*nrows))
    if not hasattr(axs, "__len__"):
        axs = np.array((axs,))
    axs = axs.reshape((nrows, ncols))

    fns = sorted(filter(
        lambda fn: re.search(r"checkpoint(\d+)\.pt", fn) is not None,
        os.listdir(save_dir)
    ))[-axs.size:]
    for i, (fn, ax) in enumerate(zip(fns, axs.flatten())):
        _, model, optimizer, _, _ = loadcheckpoint(f"{exp_dir}/{fn}", device="cuda")
        model.eval()
        approx = approx_func(optimizer).to("cpu")
        hess_diag = torch.load(f"{save_dir}/{fn}", map_location="cpu")
        ckpt = re.search(r"checkpoint(\d+)\.pt", fn).group(1)
        plot(
            ax, x=approx.numpy(), y=hess_diag.numpy(),
            checkpoint=ckpt, label_data=(i==0)
        )

    for ax in axs[:,0]:
        ax.set_ylabel("Hessian Diagonal", fontsize=20)
    for ax in axs[-1,:]:
        ax.set_xlabel("Approximation", fontsize=20)

    fig.tight_layout()
    plt.savefig(f"{save_dir}/{approx_func.__name__}.png")


if __name__ == "__main__":

    DATA_SAMPLES = 5000
    HUTCHINSON_SAMPLES = 250
    NROWS, NCOLS = 2, 3

    EXP_DIR = "results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-10-34-12"
    APPROX_FUNC = ivon_hess
    main(EXP_DIR, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, NROWS, NCOLS)