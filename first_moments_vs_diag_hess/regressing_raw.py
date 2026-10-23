import os

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from constants import ASSETS
from main import parser, tracker_fns_map, run
from utilities import power_law_offset_format


def power_law_offset(x, a=1., b=1., c=1.):
    return a + b * (x ** c)


def plot(xs, ys, ckpts, xlabel, ylabel, log_every, save_fn=None):

    fig, axs = plt.subplots(2, 3, figsize=(7.5*3, 4.5*2))
    for i, (ckpt, ax) in enumerate(zip(ckpts, axs.flatten())):
        title = f"Checkpoint {ckpt*log_every}"
        x, y = np.asarray(xs[ckpt]), np.asarray(ys[ckpt])
        mask = (x > 1e-24) & (y > 1e-24)
        x, y = x[mask], y[mask]
        ax.scatter(x, y, s=1, color="yellowgreen", label="Data" if i==0 else None)
        if sum(mask) >= 2:
            p0 = [0., 75., 0.5]
            bounds = (0, np.inf)
            opt, _ = curve_fit(power_law_offset, x, y, p0=p0, bounds=bounds, maxfev=10000)
            x_fit = np.geomspace(x.min(), x.max(), 200)
            y_fit = power_law_offset(x_fit, *opt)
            ax.plot(x_fit, y_fit, linewidth=5, linestyle="--", color="green", label=power_law_offset_format(*opt))
            ax.legend(fontsize=20, framealpha=1, markerscale=8)
            r2 = r2_score(y, power_law_offset(x, *opt))
            title = f"{title}, $R^2 = {r2:.3f}$"
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.set_title(title, fontsize=20)
    for ax in axs[-1,:]:
        ax.set_xlabel(xlabel, fontsize=20)
    for ax in axs[:,0]:
        ax.set_ylabel(ylabel, fontsize=20)

    fig.tight_layout()
    if save_fn is not None:
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        plt.savefig(save_fn)
    plt.close()


def driver(cmd: str, ckpts: list, save_fn: str) -> None:
    args = parser.parse_args(cmd.split())
    _, out = run(args)
    tracked_vals = out[1:]
    labels = [tracker_fns_map[tracker][1] for tracker in args.trackers]
    log_every = args.steps // args.ckpts
    plot(*tracked_vals, ckpts, *labels, log_every, save_fn=save_fn)


if __name__ == "__main__":

    cmd = f"--optimizer adam"
    ckpts = [1, 3, 5, 7, 9, 10]
    print(f"*** python main.py {cmd} ***")
    driver(cmd, ckpts, f"{ASSETS}/implementation-checks/regressing-raw.png")
    print()