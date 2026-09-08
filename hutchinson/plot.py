'''
python -m hutchinson.plot
'''


import os
import re
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit
import torch
import matplotlib.pyplot as plt

from common.trainutils import loadcheckpoint


def power_law_offset(x, a, b, c):
    return np.log(a + b * (x ** c))
def power_law_offset_format(a, b, c):
    mantissa, exponent = f"{a:.2e}".split("e")
    return f"$y = {mantissa} \\cdot 10^{{{exponent}}} + {b:.2f} \\cdot x^{{{c:.2f}}}$"
power_law_offset.format = power_law_offset_format

def plot(ax, x, y, checkpoint, fit_func, label_data=True):
    mask = (x > 1e-24) & (y > 1e-24)
    x, y = x[mask], y[mask]
    ax.scatter(x, y, s=1, color="cornflowerblue", label="Data" if label_data else None)
    if sum(mask) >= 2 and isinstance(fit_func, Callable):
        p0 = [0., 75., 0.5]
        bounds = (0, np.inf)
        opt, _ = curve_fit(fit_func, x, np.log(y), p0=p0, bounds=bounds, maxfev=10000)
        x_fit = np.geomspace(x.min(), x.max(), 200)
        y_fit = np.exp(power_law_offset(x_fit, *opt))
        ax.plot(x_fit, y_fit, linewidth=5, linestyle="--", color="blue", label=fit_func.format(*opt))
        ax.legend(fontsize=20, framealpha=1., markerscale=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_title(f"Checkpoint {checkpoint}", fontsize=20)

def adam_expavgsq(optimizer):
    out = torch.cat([
        # Can debias but it barely changes anything
        optimizer.state[param].get("exp_avg_sq", torch.zeros_like(param.data)).flatten()
        for group in optimizer.param_groups
        for param in group["params"]
        if param.requires_grad
    ])
    return out

def ivon_hess(optimizer):
    out = torch.cat([
        group.get("hess", torch.zeros(group["numel"]))
        for group in optimizer.param_groups
    ])
    return out

def main(exp_dir, approx_func, data_samples, hutchinson_samples, fit_func, nrows, ncols):

    save_dir = f"{exp_dir}/hutchinson/ds={data_samples}_hs={hutchinson_samples}"

    fig, axs = plt.subplots(nrows, ncols, figsize=(7.5*ncols, 4.5*nrows))
    if not hasattr(axs, "__len__"):
        axs = np.array((axs,))
    axs = axs.reshape((nrows, ncols))

    fns = sorted(os.listdir(save_dir))[-axs.size:]
    for i, (fn, ax) in enumerate(zip(fns, axs.flatten())):
        _, model, optimizer, _, _ = loadcheckpoint(f"{exp_dir}/{fn}", device="cpu")
        model.eval()
        approx = approx_func(optimizer)
        hess_diag = torch.load(f"{save_dir}/{fn}", map_location="cpu")
        ckpt = re.search(r"checkpoint(\d+)\.pt", fn).group(1)
        plot(
            ax, x=approx.numpy(), y=hess_diag.numpy(),
            checkpoint=ckpt, fit_func=fit_func, label_data=(i==0)
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

    EXP_DIR = "results/cifar10/resnet20/adam/seed=0/2026-09-02-17-15-51"
    APPROX_FUNC = adam_expavgsq
    FIT_FUNC = power_law_offset
    main(EXP_DIR, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, FIT_FUNC, NROWS, NCOLS)

    EXP_DIR = "results/cifar10/resnet20/adamw/seed=0/2026-08-27-21-41-19"
    APPROX_FUNC = adam_expavgsq
    FIT_FUNC = power_law_offset
    main(EXP_DIR, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, FIT_FUNC, NROWS, NCOLS)

    EXP_DIR = "results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-10-34-12"
    APPROX_FUNC = ivon_hess
    FIT_FUNC = None
    main(EXP_DIR, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, FIT_FUNC, NROWS, NCOLS)

    # EXP_DIR = "results/cifar10/resnet20/ivadam-decoupled-atmean/seed=0/2026-09-02-16-19-40"
    # APPROX_FUNC = adam_expavgsq
    # FIT_FUNC = power_law_offset
    # main(EXP_DIR, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, FIT_FUNC, NROWS, NCOLS)

    # EXP_DIR = "results/cifar10/resnet20/ivadam-decoupled/seed=0/2026-08-27-21-41-17"
    # APPROX_FUNC = adam_expavgsq
    # FIT_FUNC = power_law_offset
    # main(EXP_DIR, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, FIT_FUNC, NROWS, NCOLS)