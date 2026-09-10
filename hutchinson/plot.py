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
    APPROX_FUNC = adam_expavgsq
    FIT_FUNC = power_law_offset

    EXP_DIRS = [
        "results/cifar10/resnet20/adam/seed=0/2026-09-02-17-15-51",                     # Adam with AdamW (default) settings as in IVON paper
        "results/cifar10/resnet20/adam/seed=0/2026-09-08-16-35-55",                     # Adam with \beta_2 = 0.95
        "results/cifar10/resnet20/adam/seed=0/2026-09-09-16-11-04",                     # Adam with cosine decay to \eta_{\min} = 0.1 * \eta_{\max}
        "results/cifar10/resnet20/adam/seed=1/2026-09-09-22-11-39",
        "results/cifar10/resnet20/adam/seed=2/2026-09-09-22-12-17",
        "results/cifar10/resnet20/adam/seed=3/2026-09-09-22-12-19",
        "results/cifar10/resnet20/adam/seed=4/2026-09-09-22-12-21",
        "results/cifar10/resnet20/adamw/seed=0/2026-08-27-21-41-19",                    # AdamW with IVON paper (default) settings
        "results/cifar10/resnet20/adamw/seed=0/2026-09-08-16-35-50",                    # AdamW with \beta_2 = 0.95
        "results/cifar10/resnet20/adamw/seed=0/2026-09-09-16-11-13",                    # AdamW with cosine decay to \eta_{\min} = 0.1 * \eta_{\max}
        "results/cifar10/resnet20/adamw/seed=1/2026-09-10-13-11-51",
        "results/cifar10/resnet20/adamw/seed=2/2026-09-10-13-12-26",
        "results/cifar10/resnet20/adamw/seed=3/2026-09-10-13-12-27",
        "results/cifar10/resnet20/adamw/seed=4/2026-09-10-13-12-30",
        "results/cifar10/resnet20/ivadam-coupled-atmean/seed=0/2026-09-08-11-35-14",    # VAdam with AdamW settings in IVON paper
        "results/cifar10/resnet20/ivadam-decoupled-atmean/seed=0/2026-09-02-16-19-40",  # VAdam, but with weight decay not included in first-moment -- not truly decoupled
        "results/cifar10/resnet20/ivadam-decoupled/seed=0/2026-08-27-21-41-17",         # VAdam with decoupled weight decay and sampling
    ]

    for EXP_DIR in EXP_DIRS:
        main(EXP_DIR, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, FIT_FUNC, NROWS, NCOLS)

    # IVON-Price
    EXP_DIR = "results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-10-34-12"
    main(EXP_DIR, ivon_hess, DATA_SAMPLES, HUTCHINSON_SAMPLES, None, NROWS, NCOLS)