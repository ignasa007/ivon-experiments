import os
from os.path import basename, splitext
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


setups = [
    ([
        "results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-10-34-12",
        "results/cifar10/resnet20/ivon-price/seed=1/2026-09-06-23-21-19",
        "results/cifar10/resnet20/ivon-price/seed=2/2026-09-06-23-21-11",
        "results/cifar10/resnet20/ivon-price/seed=3/2026-09-06-23-21-25",
        "results/cifar10/resnet20/ivon-price/seed=4/2026-09-06-23-21-38",
    ], "IVON-Price"),
    ([
        "results/cifar10/resnet20/ivon-gradsq/seed=0/2026-09-06-23-23-29",
        "results/cifar10/resnet20/ivon-gradsq/seed=3/2026-09-07-01-42-24",
        "results/cifar10/resnet20/ivon-gradsq/seed=4/2026-09-07-01-35-29",
        "results/cifar10/resnet20/ivon-gradsq/seed=2/2026-09-07-01-37-38",
        "results/cifar10/resnet20/ivon-gradsq/seed=1/2026-09-07-01-34-47",
    ], "IVON-Gradsq"),
]
linestyles = ["-", "-.", "--", ":"]
col = "acc"

fig, axs = plt.subplots(1, 2, figsize=(7.2*2, 4.8))

for ax, split in zip(axs, ("train", "test")):
    for (dirs, name), linestyle in zip(setups, linestyles):
        metrics = list()
        for d in dirs:
            fn = f"{d}/{split}.csv"
            df = pd.read_csv(fn, header=0, index_col=False)
            label = name if name is not None else basename(d)
            metrics.append(df.loc[:, col])
        mean, std = np.mean(metrics, axis=0), np.std(metrics, ddof=1, axis=0)
        p = ax.plot(df.loc[:, "epoch"], mean, linestyle=linestyle, linewidth=5, label=label)
        ax.fill_between(df.loc[:, "epoch"], mean-3*std, mean+3*std, color=p[-1].get_color(), alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_title(f"{split.capitalize()}ing Set", fontsize=16)
    ax.grid()

axs[0].set_ylabel("Accuracy", fontsize=16)
axs[0].legend(fontsize=16, framealpha=1.0)
fig.tight_layout()
plt.savefig(f"{splitext(basename(inspect.stack()[0][1]))[0]}_{col}.png")