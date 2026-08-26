from os.path import basename, splitext
import inspect

import pandas as pd
import matplotlib.pyplot as plt


dirs = [
    # Algo 1 in https://arxiv.org/pdf/2402.17641
    ("results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-10-34-12", None),
    # Removed Riemannian GD term (line 5, in red)   
    ("results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-11-38-47", None),
    # Removed second term in line 5, based on the assumption that ~1 β2 => update is h <- β2 h
    ("results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-11-39-50", None),
    # Removed δ from sampling stdev, step-size rescaling, and denom in lines in 7 and 8 -- idea being it is << h
    ("results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-11-57-06", None),
]
split = "train"
col = "acc"


fig, axs = plt.subplots(1, 1, figsize=(6.4,4.8))

for d, name in dirs:
    fn = f"{d}/{split}.csv"
    df = pd.read_csv(fn, header=0, index_col=False)
    label = name if name is not None else basename(d)
    axs.plot(df.loc[:, "epoch"], df.loc[:, col], linewidth=3, label=label)

axs.tick_params(axis="both", which="major", labelsize=12)
axs.set_ylim(0.8, None)
axs.set_xlabel("Epoch", fontsize=16)
axs.set_ylabel(col.capitalize(), fontsize=16)
axs.grid()
axs.legend(fontsize=12, framealpha=1.0)

fig.tight_layout()
plt.savefig(f"{splitext(basename(inspect.stack()[0][1]))[0]}_{split}_{col}.png")