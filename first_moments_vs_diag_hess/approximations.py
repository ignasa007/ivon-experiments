from tqdm import trange
import numpy as np
from scipy.optimize import curve_fit
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt

from constants import *
from utilities import MLP, train_gd, loss_fn, \
    compute_gradient, compute_exp_avg_sq, compute_hess_diag, \
    power_law_offset, power_law_offset_format


def plot(ax, x, y, xlabel, ylabel, label_data=False):
    x, y = np.asarray(x), np.asarray(y)
    mask = (x > 1e-24) & (y > 1e-24)
    x, y = x[mask], y[mask]
    ax.scatter(x, y, s=1, color="yellowgreen", label="Data" if label_data else None)
    if sum(mask) >= 2:
        p0 = [0., 75., 0.5]
        bounds = (0, np.inf)
        opt, _ = curve_fit(power_law_offset, x, np.log(y), p0=p0, bounds=bounds, maxfev=10000)
        x_fit = np.geomspace(x.min(), x.max(), 200)
        y_fit = np.exp(power_law_offset(x_fit, *opt))
        ax.plot(x_fit, y_fit, linewidth=5, linestyle="--", color="green", label=power_law_offset_format(*opt))
        ax.legend(fontsize=20, framealpha=1, markerscale=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.grid()
    ax.tick_params(axis="both", which="major", labelsize=15)

if __name__ == "__main__":

    OUTPUT_DIM = 10
    MODEL, WIDTHS, MODEL_KWARGS = MLP, [64]*3, dict()
    ACTIVATION, ACT_KWARGS = "ReLU", dict()
    LOSS_TYPE = "Mean Squared Error"
    OPTIMIZER, OPTIM_KWARGS = optim.Adam, dict(lr=1e-3, betas=(0.9, 0.95))
    EPOCHS = 1000; LOG_EVERY = EPOCHS // 10

    assets, out = train_gd(
        Dataset=torchvision.datasets.MNIST, output_dim=OUTPUT_DIM, subset_size=SUBSET_SIZE,
        Model=MODEL, widths=WIDTHS, model_kwargs=MODEL_KWARGS,
        act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
        loss_type=LOSS_TYPE, Optimizer=OPTIMIZER, optim_kwargs=OPTIM_KWARGS,
        epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED, tracker_fns=(),
    )

    X, Y, model, optimizer = assets

    fig, axs = plt.subplots(1, 2, figsize=(7.5*2, 4.5*1))
    if not hasattr(axs, "__len__"):
        axs = np.atleast_2d(np.array((axs,)))
    axs_iter = iter(axs.flatten())
    save_fn = "comparing-approximations.png"

    exp_avg_sq = compute_exp_avg_sq(X, Y, model, LOSS_TYPE, optimizer, DEVICE).detach().to("cpu")
    hess_diag = compute_hess_diag(X, Y, model, LOSS_TYPE, optimizer, DEVICE, HUTCHINSON_SAMPLES).detach().to("cpu")
    plot(next(axs_iter), exp_avg_sq, hess_diag, xlabel="Exp Avg Sq", ylabel="Hessian Diagonal", label_data=True)

    emp_fisher_diag = 0.
    for i in trange(X.size(0)):
        x, y = X[[i], :], Y[[i], :]
        model.train()
        optimizer.zero_grad()
        loss_fn(model(x), y, LOSS_TYPE).backward()
        emp_fisher_diag += compute_gradient(x, y, model, LOSS_TYPE, optimizer, DEVICE).square()
    emp_fisher_diag = (emp_fisher_diag/X.size(0)).to("cpu")
    plot(next(axs_iter), emp_fisher_diag, hess_diag, xlabel="Emp Fisher Diagonal", ylabel="Hessian Diagonal", label_data=True)

    fig.tight_layout()
    if save_fn is not None:
        plt.savefig(save_fn)
    plt.show()