import os

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from utilities import power_law_offset, power_law_offset_format
from main import parser, tracker_fns_map, run


CMD = "--optimizer adam --widths [32]*1 --steps 1000 --ckpts 5 --trackers exp_avg_sq exact_ggn_diag exact_hess_diag"
SAVE_FN = "assets/adam/expavgsq-ggn-hess.png"
CKPTS = [1, 3, 5]

args = parser.parse_args(CMD.split())
_, out = run(args)
tracked_vals = out[1:]
labels = [tracker_fns_map[tracker][1] for tracker in args.trackers]

fig, axs = plt.subplots(len(CKPTS)-1, 3, figsize=(7.5*3, 4.5*(len(CKPTS)-1)))
log_every = args.steps // args.ckpts

for i, (xs, ys, xlabel, ylabel, axs_row) in enumerate(zip(
    tracked_vals[:-1], tracked_vals[1:], labels[:-1], labels[1:], axs
)):
    
    # Need ylabel only for first column cause shared with each row
    axs_row[0].set_ylabel(ylabel, fontsize=20)

    for j, (ckpt, ax) in enumerate(zip(CKPTS, axs_row)):

        # Checkpoint in the title only in top row
        title = f"Checkpoint {ckpt*log_every}"
        x, y = np.asarray(xs[ckpt]), np.asarray(ys[ckpt])
        mask = (x > 1e-24) & (y > 1e-24)
        x, y = x[mask], y[mask]
        ax.scatter(x, y, s=1, color="yellowgreen", label="Data" if i==j==0 else None)

        if sum(mask) >= 2:
            p0 = [0., 75., 0.5]
            bounds = (0, np.inf)
            opt, _ = curve_fit(power_law_offset, x, np.log(y), p0=p0, bounds=bounds, maxfev=10000)
            x_fit = np.geomspace(x.min(), x.max(), 200)
            y_fit = np.exp(power_law_offset(x_fit, *opt))
            ax.plot(x_fit, y_fit, linewidth=5, linestyle="--", color="green", label=power_law_offset_format(*opt))
            ax.legend(fontsize=20, framealpha=1, markerscale=8)
            r2 = r2_score(np.log(y), power_law_offset(x, *opt))
            title = f"{title}, $R^2 = {r2:.3f}$"
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
        ax.tick_params(axis="both", which="major", labelsize=15)
        # Need xlabels for all because not shared across rows 
        ax.set_xlabel(xlabel, fontsize=20)
        # Need title for all plots because we have R2 scores
        ax.set_title(title, fontsize=20)

fig.tight_layout()
if SAVE_FN is not None:
    os.makedirs(os.path.dirname(SAVE_FN), exist_ok=True)
    plt.savefig(SAVE_FN)
plt.close()