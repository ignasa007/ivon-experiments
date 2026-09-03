from os.path import basename, splitext
import inspect

import pandas as pd
import matplotlib.pyplot as plt


dirs = [
    # Algo 1 in https://arxiv.org/pdf/2402.17641
    ("results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-10-34-12", "IVON-Price"),
    # # Removed Riemannian GD term (line 5, in red)
    # ("results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-11-38-47", None),
    # # Removed second term in line 5, based on the assumption that ~1 β2 => update is h <- β2 h
    # ("results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-11-39-50", None),
    # # Removed δ from sampling stdev, step-size rescaling, and denom in lines in 7 and 8 -- idea being it is << h
    # ("results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-11-57-06", None),
    # AdamW with IVON-paper settings
    ("results/cifar10/resnet20/adamw/seed=0/2026-08-27-21-41-19", "AdamW"),
    # IVAdam with step-size 2e-3
    ("results/cifar10/resnet20/ivadam-decoupled/seed=0/2026-08-27-21-41-17", r"IVAdam ($\eta=2\cdot 10^{-3}, \delta=2\cdot 10^{-4}$)"),
    # # IV Adam with step-size 2e-2
    # ("results/cifar10/resnet20/ivadam-decoupled/seed=0/2026-08-27-23-29-39", r"IVAdam ($\eta=2\cdot 10^{-2}, \delta=2\cdot 10^{-4}$)"),
    # # IV Adam with step-size 2e-1
    # ("results/cifar10/resnet20/ivadam-decoupled/seed=0/2026-08-27-23-30-31", r"IV Adam with $\eta=0.2$"),
    # Adam with AdamW settings
    ("results/cifar10/resnet20/adam/seed=0/2026-09-02-17-15-51", "Adam"),
    # IVAdam without sampling
    ("results/cifar10/resnet20/ivadam-decoupled-atmean/seed=0/2026-09-02-16-19-40", "IVAdam@mean"),
]
col = "acc"

fig, axs = plt.subplots(1, 2, figsize=(7.2*2, 4.8))

for ax, split in zip(axs, ("train", "test")):
    for d, name in dirs:
        fn = f"{d}/{split}.csv"
        df = pd.read_csv(fn, header=0, index_col=False)
        label = name if name is not None else basename(d)
        ax.plot(df.loc[:, "epoch"], df.loc[:, col], linewidth=3, label=label)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.grid()

axs[0].set_ylabel("Accuracy", fontsize=16)
axs[0].legend(fontsize=16, framealpha=1.0)
fig.tight_layout()
plt.savefig(f"{splitext(basename(inspect.stack()[0][1]))[0]}_{col}.png")