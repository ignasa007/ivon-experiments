import os
from functools import partial

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


def fit_arbitrary_parameters(size, base_noise_sampler, deltas, mc_samples):

    a = np.exp(np.random.uniform(-10, 0, size))
    A = np.diag(a)
    log_mod_a = np.log(np.abs(a))

    theoretical_opts, empirical_opts = list(), list()

    for delta in tqdm(deltas):

        b = np.random.uniform(-5, 5, (size, 1))
        c = 10 * np.random.randn(1).item()

        noise = delta * base_noise_sampler(size=size*mc_samples).reshape(size, mc_samples)
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

    return tuple(map(
        lambda a: np.array(a).reshape(deltas.size, mc_samples, 2),
        (theoretical_opts, empirical_opts)
    ))

def plot_arbitrary_parameters(save_fn, sizes, base_noise_sampler, deltas, mc_samples):

    sizes = sorted(sizes)
    out = [fit_arbitrary_parameters(size, base_noise_sampler, deltas, mc_samples) for size in sizes]

    fig, axs = plt.subplots(1, 3, figsize=(6.4*3, 4.8*1))
    cmap = "managua"
    xs = 0.0, 1.0
    ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    for i, j in enumerate((0, -1)):
        ax, size, (theoretical_opts, empirical_opts) = axs[i], sizes[j], out[j]
        # Permuting because otherwise the color for \delta = 1 overlays all others
        perm = np.random.permutation(deltas.size*mc_samples)
        theoretical_gammas = theoretical_opts.reshape(-1, 2)[perm, 1]
        empirical_gammas = empirical_opts.reshape(-1, 2)[perm, 1]
        ax.plot(xs, xs, linewidth=2, linestyle="--", color="black", zorder=1)
        colors = np.repeat(deltas, mc_samples)[perm]
        scatter = ax.scatter(empirical_gammas, theoretical_gammas, c=colors, cmap=cmap, s=100)
        ax.set_xlim(*xs); ax.set_ylim(*xs)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.set_xlabel(rf"Empirical $\gamma^{{\star}}$", fontsize=20)
        ax.set_ylabel(rf"Theoretical $\gamma^{{\star}}$", fontsize=20)
        ax.set_title(rf"$d = {size}$", fontsize=20)
        ax.grid()

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Sampling $\\delta$", fontsize=20, rotation=270, labelpad=28)
    cbar.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])    # TODO: hard-coded
    cbar.ax.tick_params(labelsize=15)

    ys = list()
    for size, (theoretical_opts, empirical_opts) in zip(sizes, out):
        empirical_gammas = empirical_opts[..., 1]
        ys.append(np.var(empirical_gammas.flatten()))
    axs[2].plot(sizes, ys, linewidth=5, marker="o", markersize=10)
    axs[2].set_xlabel("$d$", fontsize=20)
    axs[2].set_ylabel(r"Var$[\gamma^{\star}]$", fontsize=20)
    axs[2].set_xscale("log")
    axs[2].tick_params(axis="both", which="major", labelsize=15)
    axs[2].grid()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_fn), exist_ok=True)
    plt.savefig(save_fn)
    plt.close()


def noise_sampler_titles(noise_sampler_name):
    if noise_sampler_name == "uniform":
        return "Uniform Distribution"
    elif noise_sampler_name == "normal":
        return "Normal Distribution"

def theoretical_opts_random_perturbations(log_mod_a, delta, noise_sampler_name="uniform"):
    if noise_sampler_name == "uniform":
        mean, var = -1, 1
    elif noise_sampler_name == "normal":
        mean, var = -0.6351, 1.2377
    gamma_opt = 0.5 * (1 + var/np.var(log_mod_a)) ** (-1)
    beta_opt = np.exp(
        (1-2*gamma_opt) * np.mean(log_mod_a) - 2*gamma_opt * (np.log(delta)+mean)
    )
    return beta_opt, gamma_opt

def fit_random_perturbations(size, noise_sampler_name, base_noise_sampler, deltas, mc_samples):

    a = np.exp(np.random.uniform(-10, 0, size))
    A = np.diag(a)
    log_mod_a = np.log(np.abs(a))

    theoretical_opts, empirical_opts = list(), list()

    for delta in tqdm(deltas):

        b = np.random.uniform(-5, 5, (size, 1))
        c = 10 * np.random.randn(1).item()

        theoretical_opts.append(theoretical_opts_random_perturbations(log_mod_a, delta, noise_sampler_name))
        noise = delta * base_noise_sampler(size=size*mc_samples).reshape(size, mc_samples)
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

    return theoretical_opts, empirical_opts

def plot_random_perturbation(save_fn, size, noise_sampler_names, base_noise_samplers, deltas, mc_samples):

    fig, axs = plt.subplots(1, 3, figsize=(6.4*3, 4.8*1))

    base_noise_sampler, noise_sampler_name = base_noise_samplers[0], noise_sampler_names[0]
    theoretical_opts, empirical_opts = fit_random_perturbations(
        size, noise_sampler_name, base_noise_sampler, deltas, mc_samples
    )
    theoretical_betas, theoretical_gammas = zip(*theoretical_opts)
    empirical_betas, empirical_gammas = zip(*empirical_opts)

    axs[0].plot(
        deltas, theoretical_betas, marker="o", markersize=10,
        linewidth=5, color="black", label=rf"Theoretical $\beta^{{\star}}$"
    )
    axs[0].plot(
        deltas, empirical_betas, marker="o", markersize=10,
        linewidth=5, linestyle="--", label=rf"Empirical $\beta^{{\star}}$"
    )
    axs[0].set_ylim(bottom=0)
    axs[0].set_title(noise_sampler_titles(noise_sampler_name), fontsize=20)
    
    axs[1].plot(
        deltas, theoretical_gammas, marker="o", markersize=10,
        linewidth=5, color="black", label=rf"Theoretical $\gamma^{{\star}}$"
    )
    axs[1].plot(
        deltas, empirical_gammas, marker="o", markersize=10,
        linewidth=5, linestyle="--", label=rf"Empirical $\gamma^{{\star}}$"
    )
    axs[1].set_ylim(bottom=0, top=1)
    axs[1].set_title(noise_sampler_titles(noise_sampler_name), fontsize=20)

    base_noise_sampler, noise_sampler_name = base_noise_samplers[1], noise_sampler_names[1]
    theoretical_opts, empirical_opts = fit_random_perturbations(
        size, noise_sampler_name, base_noise_sampler, deltas, mc_samples
    )
    theoretical_betas, theoretical_gammas = zip(*theoretical_opts)
    empirical_betas, empirical_gammas = zip(*empirical_opts)

    axs[2].plot(
        deltas, theoretical_gammas, marker="o", markersize=10,
        linewidth=5, color="black", label=rf"Theoretical $\gamma^{{\star}}$"
    )
    axs[2].plot(
        deltas, empirical_gammas, marker="o", markersize=10,
        linewidth=5, linestyle="--", label=rf"Empirical $\gamma^{{\star}}$"
    )
    axs[2].set_ylim(bottom=0, top=1)
    axs[2].set_title(noise_sampler_titles(noise_sampler_name), fontsize=20)

    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.set_xlabel(rf"$\delta$", fontsize=20)
        ax.grid()
        ax.legend(framealpha=1, fontsize=20)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_fn), exist_ok=True)
    plt.savefig(save_fn)
    plt.close()


if __name__ == "__main__":

    np.random.seed(0)

    save_fn = f"{ASSETS}/theory/arbitrary-parameters-fit.png"
    sizes = (10, 100, 1000)
    base_noise_sampler = partial(np.random.uniform, low=-1, high=1)
    deltas = np.linspace(0.1, 1, 10)
    mc_samples = 100
    plot_arbitrary_parameters(save_fn, sizes, base_noise_sampler, deltas, mc_samples)

    save_fn = f"{ASSETS}/theory/random-perturbation-fit.png"
    size = 100
    base_noise_samplers = (
        partial(np.random.uniform, low=-1, high=1),
        lambda size: np.random.randn(size)              # Need a common API for both samplers
    )
    noise_sampler_names = ("uniform", "normal")
    deltas = np.linspace(0.1, 1, 10)
    mc_samples = 100
    plot_random_perturbation(save_fn, size, noise_sampler_names, base_noise_samplers, deltas, mc_samples)