import os

from tqdm import tqdm
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from constants import ASSETS


def assert_type(x: np.ndarray, A: np.ndarray, b: np.ndarray, c: float):
    assert isinstance(x, np.ndarray) and isinstance(A, np.ndarray) \
        and isinstance(b, np.ndarray) and isinstance(c, float), \
        f"{type(x) = }, {type(A) = }, {type(b) = }, {type(c) = }"

def assert_shapes(x: np.ndarray, A: np.ndarray, b: np.ndarray):
    assert x.ndim == A.ndim == b.ndim == 2 \
        and A.shape[0] == A.shape[1] == x.shape[0] == b.shape[0] \
        and b.shape[1] == 1, \
        f"{x.shape = }, {A.shape = }, {b.shape = }"
    
def assert_inputs(x: np.ndarray, A: np.ndarray, b: np.ndarray, c: float):
    if x is None:
        x = b
    assert_type(x, A, b, c)
    assert_shapes(x, A, b)

def loss(x: np.ndarray, A: np.ndarray, b: np.ndarray, c: float = 0.) -> float:
    assert_inputs(x, A, b, c)
    return (0.5 * x.T @ A @ x - x.T @ b + c).item()

def grad(x: np.ndarray, A: np.ndarray, b: np.ndarray, c: float = 0.) -> np.ndarray:
    assert_inputs(x, A, b, c)
    return A @ x - b

def hess_diag(x: np.ndarray, A: np.ndarray, b: np.ndarray, c: float = 0.) -> np.ndarray:
    assert_inputs(x, A, b, c)
    return np.repeat(np.diag(A)[:, None], x.shape[1], axis=1)

def minima(A: np.ndarray, b: np.ndarray, c: float = 0.) -> np.ndarray:
    assert_inputs(None, A, b, c)
    return np.linalg.pinv(A) @ b

def power_law(x, beta=1., gamma=1.):
    return np.log(beta * (x ** gamma))


if __name__ == "__main__":

    size = 10
    mc_samples = 100
    unif_deltas = np.linspace(0.1, 1, 10)
    np.random.seed(0)

    a = np.exp(np.random.uniform(-10, 0, size))
    A = np.diag(a)
    log_mod_a = np.log(np.abs(a))

    save_fn = f"{ASSETS}/theory/log-space-arbitrary-param.png"
    fig, axs = plt.subplots(1, 1, figsize=(6.4*1, 4.8*1))
    theoretical_opts, empirical_opts = list(), list()

    for unif_delta in tqdm(unif_deltas):

        b = np.random.uniform(-5, 5, (size, 1))
        c = 10 * np.random.randn(1).item()

        noise = np.random.uniform(-unif_delta, +unif_delta, (size, mc_samples))
        points = minima(A, b, c) + noise
        grads = grad(points, A, b, c)
        hess_diags = hess_diag(points, A, b, c)

        log_mod_grads = np.log(np.abs(grads))
        gamma_opts = 0.5 * (
            np.cov(log_mod_a, log_mod_grads, bias=True, rowvar=False)[0, 1:] \
            / np.var(log_mod_grads, axis=0)
        )
        beta_opts = np.exp(np.mean(log_mod_a) - 2*gamma_opts * np.mean(log_mod_grads))
        theoretical_opts.extend(list(zip(beta_opts, gamma_opts)))

        for g, h in zip(grads.T, hess_diags.T):
            p0 = [75., 0.5]
            bounds = (0, np.inf)
            opt, _ = curve_fit(
                power_law, np.square(g), np.log(h),
                p0=p0, bounds=bounds, maxfev=10000
            )
            empirical_opts.append(opt)

    theoretical_betas, theoretical_gammas = zip(*theoretical_opts)
    empirical_betas, empirical_gammas = zip(*empirical_opts)

    axs.scatter(empirical_gammas, theoretical_gammas)
    axs.set_xlim(0.15, 0.75)
    axs.set_ylim(0.15, 0.75)
    axs.tick_params(axis="both", which="major", labelsize=15)
    axs.set_xlabel(rf"Empirical $\gamma^{{\star}}$", fontsize=20)
    axs.set_ylabel(rf"Theoretical $\gamma^{{\star}}$", fontsize=20)
    axs.grid()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_fn), exist_ok=True)
    plt.savefig(save_fn)
    plt.close()

    save_fn = f"{ASSETS}/theory/log-space-random-pert.png"
    fig, axs = plt.subplots(1, 2, figsize=(6.4*2, 4.8*1))
    theoretical_opts, empirical_opts = list(), list()

    for unif_delta in tqdm(unif_deltas):

        b = np.random.uniform(-5, 5, (size, 1))
        c = 10 * np.random.randn(1).item()

        gamma_opt = 0.5 * (1 + 1/np.var(log_mod_a)) ** (-1)
        beta_opt = np.exp(
            (1-2*gamma_opt) * np.mean(log_mod_a) - 2*gamma_opt * (np.log(unif_delta)-1)
        )
        theoretical_opts.append((beta_opt, gamma_opt))

        noise = np.random.uniform(-unif_delta, +unif_delta, (size, mc_samples))
        points = minima(A, b, c) + noise
        grads = grad(points, A, b, c).flatten()
        hess_diags = hess_diag(points, A, b, c).flatten()

        p0 = [75., 0.5]
        bounds = (0, np.inf)
        opt, _ = curve_fit(
            power_law, np.square(grads), np.log(hess_diags),
            p0=p0, bounds=bounds, maxfev=10000
        )
        empirical_opts.append(opt)

    theoretical_betas, theoretical_gammas = zip(*theoretical_opts)
    empirical_betas, empirical_gammas = zip(*empirical_opts)

    axs[0].plot(
        unif_deltas, empirical_betas, marker="o", markersize=10,
        linewidth=5, label=rf"Empirical $\beta^{{\star}}$"
    )
    axs[0].plot(
        unif_deltas, theoretical_betas, marker="o", markersize=10,
        linewidth=5, linestyle="--", color="black", label=rf"Theoretical $\beta^{{\star}}$"
    )
    axs[0].set_ylim(bottom=0)
    axs[1].plot(
        unif_deltas, empirical_gammas, marker="o", markersize=10,
        linewidth=5, label=rf"Empirical $\gamma^{{\star}}$"
    )
    axs[1].plot(
        unif_deltas, theoretical_gammas, marker="o", markersize=10,
        linewidth=5, linestyle="--", color="black", label=rf"Theoretical $\gamma^{{\star}}$"
    )
    axs[1].set_ylim(bottom=0, top=1)

    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.set_xlabel(rf"$\delta$", fontsize=20)
        ax.grid()
        ax.legend(framealpha=1, fontsize=20)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_fn), exist_ok=True)
    plt.savefig(save_fn)
    plt.close()