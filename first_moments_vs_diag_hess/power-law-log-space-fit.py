import os

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

    size = 100
    mc_samples = 100
    unif_deltas = np.linspace(0.1, 1, 10)
    save_fn = f"{ASSETS}/theory/power-law-log-space-fit.png"

    a = np.exp(np.random.uniform(-10, 0, size))
    A = np.diag(a)

    fig, axs = plt.subplots(1, 2, figsize=(6.4*2, 4.8*1))
    theoretical_opts, empirical_opts = list(), list()

    for unif_delta in unif_deltas:

        b = np.random.uniform(-5, 5, (size, 1))
        c = 10 * np.random.randn(1).item()

        log_mod_a = np.log(np.abs(a))
        gamma_opt = 0.5 * (1 + 1/np.var(log_mod_a)) ** (-1)
        beta_opt = np.exp(
            (1-2*gamma_opt) * np.mean(log_mod_a) - 2*gamma_opt * (np.log(unif_delta)-1)
        )
        theoretical_opts.append((beta_opt, gamma_opt))

        noise = np.random.uniform(-unif_delta, +unif_delta, (size, mc_samples))
        points = minima(A, b, c) + noise
        grad_sqs = np.square(grad(points, A, b, c)).flatten()
        hess_diags = hess_diag(points, A, b, c).flatten()

        p0 = [75., 0.5]
        bounds = (0, np.inf)
        opt, _ = curve_fit(
            power_law, grad_sqs, np.log(hess_diags),
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