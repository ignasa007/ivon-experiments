from os.path import basename, splitext
import inspect
import math

import pandas as pd
import matplotlib.pyplot as plt


dirs = [
    ("results/cifar10/resnet20/adam/seed=0/2026-09-02-17-15-51", "Adam"),
    ("results/cifar10/resnet20/adam/seed=0/2026-09-08-16-35-55", r"Adam ($\beta_2=0.95$)"),
    ("results/cifar10/resnet20/adam/seed=0/2026-09-09-16-11-04", r"Adam ($\eta_{\min}=0.1\cdot\eta_{\max}$)"),
    ("results/cifar10/resnet20/adamw/seed=0/2026-08-27-21-41-19", "AdamW"),
    ("results/cifar10/resnet20/adamw/seed=0/2026-09-08-16-35-50", r"AdamW ($\beta_2=0.95$)"),
    ("results/cifar10/resnet20/adamw/seed=0/2026-09-08-16-35-50", r"AdamW ($\eta_{\min}=0.1\cdot\eta_{\max}$)"),
    ("results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-10-34-12", "IVON-Price"),
    ("results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-11-38-47", "W/o Riemannian Correction"),
    ("results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-11-39-50", "W/o Curvature Accumulation"),
    ("results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-11-57-06", "W/o WD in ELBO Curvature"),
    ("results/cifar10/resnet20/ivadam-decoupled-atmean/seed=0/2026-09-02-16-19-40", "VAdam@mean"),
    ("results/cifar10/resnet20/ivadam-decoupled/seed=0/2026-08-27-21-41-17", r"VAdam ($\eta_{\max}=2\cdot 10^{-3}, \delta=2\cdot 10^{-4}$)"),
    ("results/cifar10/resnet20/ivadam-decoupled/seed=0/2026-08-27-23-29-39", r"VAdam ($\eta_{\max}=2\cdot 10^{-2}, \delta=2\cdot 10^{-4}$)"),
    ("results/cifar10/resnet20/ivadam-decoupled/seed=0/2026-08-27-23-30-31", r"VAdam ($\eta_{\max}=2\cdot 10^{-1}, \delta=2\cdot 10^{-4}$)"),
]

col = "acc"
linestyles = ["-", "-.", "--", ":"]
linestyles = (linestyles * math.ceil(len(dirs)/len(linestyles)))[:len(dirs)]

fig, axs = plt.subplots(1, 2, figsize=(7.2*2, 4.8))

for ax, split in zip(axs, ("train", "test")):
    for (d, name), linestyle in zip(dirs, linestyles):
        fn = f"{d}/{split}.csv"
        df = pd.read_csv(fn, header=0, index_col=False)
        label = name if name is not None else basename(d)
        ax.plot(df.loc[:, "epoch"], df.loc[:, col], linestyle=linestyle, linewidth=5, label=label)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_title(f"{split.capitalize()}ing Set", fontsize=16)
    ax.grid()

axs[0].set_ylabel("Accuracy", fontsize=16)
axs[0].legend(fontsize=16, framealpha=1.0)
fig.tight_layout()
plt.savefig(f"{splitext(basename(inspect.stack()[0][1]))[0]}_{col}.png")