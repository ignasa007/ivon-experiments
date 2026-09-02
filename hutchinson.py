import os
import math
import re
from typing import Callable

from tqdm import tqdm, trange
import numpy as np
from scipy.optimize import curve_fit
import torch
from torch import autograd
from torch.nn.utils import parameters_to_vector
import matplotlib.pyplot as plt

from common.trainutils import loadcheckpoint
from common.dataloaders import TRAINDATALOADERS


def compute_hvp(X, Y, model, vector):
    model.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(X), Y)
    grad = autograd.grad(loss, model.parameters(), create_graph=True)
    dot = parameters_to_vector(grad).mul(vector).sum()
    hvp = autograd.grad(dot, model.parameters())
    hvp = parameters_to_vector([v.contiguous() for v in hvp])
    return hvp

def compute_hess_diag(train_loader, model, data_samples, hutchinson_samples):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if data_samples > len(train_loader.dataset):
        print(
            f"data_samples={data_samples} > len(dataset)={len(train_loader.dataset)}; "
            f"using data_samples={len(train_loader.dataset)}"
        )
        data_samples = len(train_loader.dataset)
    remaining_samples = data_samples
    sum_samples = 0.
    for X, Y in tqdm(train_loader, total=math.ceil(round(data_samples/train_loader.batch_size, 8))):
        X, Y = map(lambda t: t[:remaining_samples].to(torch.device("cuda"), non_blocking=True), (X, Y))
        batch_weight = len(X) / data_samples
        for _ in range(hutchinson_samples):
            rand_vector = torch.randn(num_params, device="cuda")
            # Rademacher variables have a lower variance than normal variables,
            # allowing the use of smaller `hutchinson_samples`
            # rand_vector = torch.randint(2, size=(num_params,), device="cuda", dtype=X.dtype).mul(2).sub(1)
            sum_samples += batch_weight * rand_vector * compute_hvp(X, Y, model, rand_vector)
        remaining_samples -= len(X)
        if remaining_samples <= 0:
            break
    hess_diag = sum_samples / hutchinson_samples
    return hess_diag

def power_law_offset(x, a, b, c):
    return np.log(a + b * (x ** c))
def power_law_offset_format(a, b, c):
    mantissa, exponent = f"{a:.2e}".split("e")
    return f"Fit: $y = {mantissa} \\cdot 10^{{{exponent}}} + {b:.2f} \\cdot x^{{{c:.2f}}}$"
power_law_offset.format = power_law_offset_format

def plot(i, ax, x, y, checkpoint, fit_func):
    mask = (x > 1e-24) & (y > 1e-24)
    x, y = x[mask], y[mask]
    ax.scatter(x, y, s=1, label="Data" if i==0 else None)
    if sum(mask) >= 2:
        p0 = [0., 75., 0.5]
        bounds = (0, np.inf)
        if isinstance(fit_func, Callable):
            opt, _ = curve_fit(fit_func, x, np.log(y), p0=p0, bounds=bounds, maxfev=10000)
            x_fit = np.geomspace(x.min(), x.max(), 200)
            y_fit = np.exp(power_law_offset(x_fit, *opt))
            ax.plot(x_fit, y_fit, color="red", linestyle="--", linewidth=3, label=fit_func.format(*opt))
            ax.legend(fontsize=16, framealpha=1., markerscale=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_title(f"Checkpoint {checkpoint}", fontsize=16)

def grad(train_loader, model, optimizer, data_samples):
    model.zero_grad()
    for X, Y in train_loader:
        X, Y = map(lambda t: t[:data_samples].to(torch.device("cuda"), non_blocking=True), (X, Y))
        batch_weight = len(X) / data_samples
        loss = batch_weight * torch.nn.functional.cross_entropy(model(X), Y)
        loss.backward()
        data_samples -= len(X)
        if data_samples <= 0:
            break
    out = torch.cat([
        param.grad.abs().flatten()
        for group in optimizer.param_groups
        for param in group["params"]
        if param.requires_grad
    ])
    return out

def adam_expavg(train_loader, model, optimizer, data_samples):
    out = torch.cat([
        optimizer.state[param].get("exp_avg", torch.zeros_like(param.data)).abs().flatten()
        for group in optimizer.param_groups
        for param in group["params"]
        if param.requires_grad
    ])
    return out

def adam_expavgsq(train_loader, model, optimizer, data_samples):
    out = torch.cat([
        # Can debias but it barely changes anything
        optimizer.state[param].get("exp_avg_sq", torch.zeros_like(param.data)).flatten()
        for group in optimizer.param_groups
        for param in group["params"]
        if param.requires_grad
    ])
    return out

def ivon_hess(train_loader, model, optimizer, data_samples):
    out = torch.cat([
        group.get("hess", torch.zeros(group["numel"]))
        for group in optimizer.param_groups
    ])
    return out

def main(d, approx_func, data_samples, hutchinson_samples, fit_func, nrows, ncols):

    fig, axs = plt.subplots(nrows, ncols, figsize=(6.4*ncols, 4.8*nrows))
    if not hasattr(axs, "__len__"):
        axs = np.array((axs,))
    axs = axs.reshape((nrows, ncols))

    train_loader, _ = TRAINDATALOADERS["cifar10"](
        data_dir="./datasets", train_val_split=1.0,
        workers=1, pin_memory=True, tbatch=2500, vbatch=50      # Batch size is memory-bound
    )

    axs_iter = iter(axs.flatten()[::-1])
    i = nrows*ncols - 1
    for fn in sorted(os.listdir(d))[::-1]:
        match = re.search(r"checkpoint(\d+)\.pt", fn)
        if not match:
            continue
        try:
            ax = next(axs_iter)
        except StopIteration:
            break
        _, model, optimizer, _, _ = loadcheckpoint(f"{d}/{fn}", device="cuda")
        model.eval()
        approx = approx_func(train_loader, model, optimizer, data_samples)
        hess_diag = compute_hess_diag(train_loader, model, data_samples, hutchinson_samples)
        plot(
            i, ax, x=approx.to("cpu").numpy(), y=hess_diag.to("cpu").numpy(),
            checkpoint=match.group(1), fit_func=fit_func
        )
        i -= 1

    for ax in axs[:,0]:
        ax.set_ylabel("Hessian Diagonal", fontsize=16)
    for ax in axs[-1,:]:
        ax.set_xlabel("Approximation", fontsize=16)

    fig.tight_layout()
    plt.savefig(f"{d}/{approx_func.__name__}-vs-hess_diag.png")


if __name__ == "__main__":

    DATA_SAMPLES = 5000
    HUTCHINSON_SAMPLES = 100
    NROWS, NCOLS = 2, 3

    D = "results/cifar10/resnet20/adamw/seed=0/2026-08-27-21-41-19"
    FIT_FUNC = power_law_offset
    APPROX_FUNC = grad
    main(D, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, FIT_FUNC, NROWS, NCOLS)
    APPROX_FUNC = adam_expavg
    main(D, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, FIT_FUNC, NROWS, NCOLS)
    APPROX_FUNC = adam_expavgsq
    main(D, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, FIT_FUNC, NROWS, NCOLS)

    D = "results/cifar10/resnet20/ivadam-decoupled/seed=0/2026-08-27-21-41-17"
    FIT_FUNC = power_law_offset
    APPROX_FUNC = grad
    main(D, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, FIT_FUNC, NROWS, NCOLS)
    APPROX_FUNC = adam_expavg
    main(D, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, FIT_FUNC, NROWS, NCOLS)
    APPROX_FUNC = adam_expavgsq
    main(D, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, FIT_FUNC, NROWS, NCOLS)

    D = "results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-10-34-12"
    APPROX_FUNC = ivon_hess
    FIT_FUNC = None
    main(D, APPROX_FUNC, DATA_SAMPLES, HUTCHINSON_SAMPLES, FIT_FUNC, NROWS, NCOLS)