from typing import List, Tuple
import os
from ast import literal_eval
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constants import ASSETS, RESULTS
from main import parser


def plot(axs, read_fn: str, cmd: str, label: str) -> None:

    assert isinstance(axs, np.ndarray)
    axs = axs.flatten()

    df = pd.read_csv(read_fn, index_col="timestamp")
    df.trackers = df.trackers.apply(literal_eval)

    args = parser.parse_args(cmd.split())
    del args.device
    row = vars(args).copy()
    matches = (
        df[list(row.keys())].eq(pd.Series(row))
        | (df[list(row.keys())].isna() & pd.Series(row).isna())
    ).all(axis=1)
    if not matches.any():
        print("No matching experiment found; skipping.")
        return

    output_fn = df.iloc[np.where(matches)[0][-1]].output_fn
    with open(f"{RESULTS}/{output_fn}", "rb") as f:
        out = pickle.load(f)
        losses, laws = out["losses"], out["laws"]

    axs[0].plot(np.arange(1, len(losses)+1), losses, linewidth=5, label=label)
    steps, a, b, c = zip(*laws)
    axs[1].plot(steps, c, linewidth=5, label=label)
    axs[2].plot(steps, b, linewidth=5, label=label)
    # axs[3].plot(steps, a, linewidth=5, label=label)


def main(cmds: List[Tuple[str, str]], steps: int, read_fn: str, save_fn: str) -> None:

    fig, axs = plt.subplots(1, 3, figsize=(6.4*3, 4.8*1))

    for cmd, label in cmds:
        print(f"*** Plotting {cmd} ***")
        plot(axs, read_fn=read_fn, cmd=cmd, label=label)

    axs[0].set_yscale("log")
    axs[0].set_ylabel("Loss", fontsize=24)
    # axs[0].legend(fontsize=21, framealpha=1.0, loc="upper right")
    axs[1].hlines([0.5], 0, steps, linewidth=5, linestyle="--", colors="black")
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel(r"Exponent, $\gamma$", fontsize=24)
    # axs[2].set_ylim(1, 1000)
    axs[2].set_yscale("log")
    axs[2].set_ylabel(r"Scaling, $\beta$", fontsize=24)
    # axs[3].set_ylabel(r"Offset, $\epsilon$", fontsize=24)

    for ax in axs:
        ax.set_xlabel("Steps", fontsize=24)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.grid()

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, fontsize=21, framealpha=1.0,
        loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=len(labels)
    )

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    os.makedirs(os.path.dirname(save_fn), exist_ok=True)
    plt.savefig(save_fn)
    plt.close()


if __name__ == "__main__":

    os.makedirs(RESULTS, exist_ok=True)
    LOG_FN = f"{RESULTS}/logs.csv"

    # Datasets
    STEPS, CKPTS = 5000, 20
    cmds = [
        (f"--optimizer adam --dataset mnist", "MNIST"),
        (f"--optimizer adam --dataset fmnist", "FashionMNIST"),
        (f"--optimizer adam --dataset cifar10", "CIFAR-10"),
        (f"--optimizer adam --dataset cifar100", "CIFAR-100"),
    ]
    cmds = [(f"{cmd} --steps {STEPS} --ckpts {CKPTS}", label) for cmd, label in cmds]
    save_fn = f"{ASSETS}/datasets.png"
    main(cmds, steps=STEPS, read_fn=LOG_FN, save_fn=save_fn)

    # Subset Sizes
    STEPS, CKPTS = 2000, 20
    cmds = [
        (f"--optimizer adam --subset_size 5000", "N = 5k"),
        (f"--optimizer adam --subset_size 10000", "N = 10k"),
        (f"--optimizer adam --subset_size 15000", "N = 15k"),
        (f"--optimizer adam --subset_size 20000", "N = 20k"),
        (f"--optimizer adam --subset_size 25000", "N = 25k"),
    ]
    cmds = [(f"{cmd} --steps {STEPS} --ckpts {CKPTS}", label) for cmd, label in cmds]
    save_fn = f"{ASSETS}/subset_sizes.png"
    main(cmds, steps=STEPS, read_fn=LOG_FN, save_fn=save_fn)

    # Widths
    STEPS, CKPTS = 1000, 20
    cmds = [
        (f"--optimizer adam --widths [64]*3", "D = 64"),
        (f"--optimizer adam --widths [128]*3", "D = 128"),
        (f"--optimizer adam --widths [256]*3", "D = 256"),
        (f"--optimizer adam --widths [512]*3", "D = 512"),
    ]
    cmds = [(f"{cmd} --steps {STEPS} --ckpts {CKPTS}", label) for cmd, label in cmds]
    save_fn = f"{ASSETS}/widths.png"
    main(cmds, steps=STEPS, read_fn=LOG_FN, save_fn=save_fn)

    # Depths
    STEPS, CKPTS = 1000, 20
    cmds = [
        (f"--optimizer adam --widths [64]*1", "L = 2"),
        (f"--optimizer adam --widths [64]*3", "L = 4"),
        (f"--optimizer adam --widths [64]*5", "L = 6"),
        (f"--optimizer adam --widths [64]*7", "L = 8"),
    ]
    cmds = [(f"{cmd} --steps {STEPS} --ckpts {CKPTS}", label) for cmd, label in cmds]
    save_fn = f"{ASSETS}/depths.png"
    main(cmds, steps=STEPS, read_fn=LOG_FN, save_fn=save_fn)

    # Activations
    STEPS, CKPTS = 1000, 20
    cmds = [
        (f"--optimizer adam --activation linear", "Linear"),
        (f"--optimizer adam --activation relu", "ReLU"),
        (f"--optimizer adam --activation leakyrelu", "LeakyReLU"),
        (f"--optimizer adam --activation tanh", "Tanh"),
        (f"--optimizer adam --activation sigmoid", "Sigmoid"),
    ]
    cmds = [(f"{cmd} --steps {STEPS} --ckpts {CKPTS}", label) for cmd, label in cmds]
    save_fn = f"{ASSETS}/activations.png"
    main(cmds, steps=STEPS, read_fn=LOG_FN, save_fn=save_fn)

    # Beta1
    STEPS, CKPTS = 2000, 20
    cmds = [
        (f"--optimizer adam --beta1 0.8", r"$\beta_1=0.8$"),
        (f"--optimizer adam --beta1 0.9", r"$\beta_1=0.9$"),
        (f"--optimizer adam --beta1 0.95", r"$\beta_1=0.95$"),
        (f"--optimizer adam --beta1 0.99", r"$\beta_1=0.99$"),
        (f"--optimizer adam --beta1 0.999", r"$\beta_1=0.999$"),
    ]
    cmds = [(f"{cmd} --steps {STEPS} --ckpts {CKPTS}", label) for cmd, label in cmds]
    save_fn = f"{ASSETS}/beta1.png"
    main(cmds, steps=STEPS, read_fn=LOG_FN, save_fn=save_fn)

    # Beta2
    STEPS, CKPTS = 2000, 20
    cmds = [
        (f"--optimizer adam --beta2 0.8", r"$\beta_2=0.8$"),
        (f"--optimizer adam --beta2 0.9", r"$\beta_2=0.9$"),
        (f"--optimizer adam --beta2 0.95", r"$\beta_2=0.95$"),
        (f"--optimizer adam --beta2 0.99", r"$\beta_2=0.99$"),
        (f"--optimizer adam --beta2 0.999", r"$\beta_2=0.999$"),
    ]
    cmds = [(f"{cmd} --steps {STEPS} --ckpts {CKPTS}", label) for cmd, label in cmds]
    save_fn = f"{ASSETS}/beta2.png"
    main(cmds, steps=STEPS, read_fn=LOG_FN, save_fn=save_fn)

    # Epsilon
    STEPS, CKPTS = 2000, 20
    cmds = [
        (f"--optimizer adam --eps 1e-8", r"$\epsilon=10^{-8}$"),
        (f"--optimizer adam --eps 1e-7", r"$\epsilon=10^{-7}$"),
        (f"--optimizer adam --eps 1e-6", r"$\epsilon=10^{-6}$"),
        (f"--optimizer adam --eps 1e-5", r"$\epsilon=10^{-5}$"),
    ]
    cmds = [(f"{cmd} --steps {STEPS} --ckpts {CKPTS}", label) for cmd, label in cmds]
    save_fn = f"{ASSETS}/eps.png"
    main(cmds, steps=STEPS, read_fn=LOG_FN, save_fn=save_fn)

    # Weight Decay
    STEPS, CKPTS = 2000, 20
    cmds = [
        (f"--optimizer adam --weight_decay 0", r"$\lambda=0$"),
        (f"--optimizer adam --weight_decay 1e-4", r"$\lambda=10^{-4}$"),
        (f"--optimizer adam --weight_decay 1e-3", r"$\lambda=10^{-3}$"),
        (f"--optimizer adam --weight_decay 1e-2", r"$\lambda=10^{-2}$"),
    ]
    cmds = [(f"{cmd} --steps {STEPS} --ckpts {CKPTS}", label) for cmd, label in cmds]
    save_fn = f"{ASSETS}/weight_decay.png"
    main(cmds, steps=STEPS, read_fn=LOG_FN, save_fn=save_fn)

    # Batch Size
    STEPS, CKPTS = 10000, 20
    cmds = [
        (f"--optimizer adam --batch_size 100", "BS = 100"),
        (f"--optimizer adam --batch_size 500", "BS = 500"),
        (f"--optimizer adam --batch_size 1000", "BS = 1k"),
        (f"--optimizer adam --batch_size 5000", "BS = 5k"),
    ]
    cmds = [(f"{cmd} --steps {STEPS} --ckpts {CKPTS}", label) for cmd, label in cmds]
    save_fn = f"{ASSETS}/batch_size.png"
    main(cmds, steps=STEPS, read_fn=LOG_FN, save_fn=save_fn)

    # Hutchinson Samples
    STEPS, CKPTS = 2000, 20
    cmds = [
        (f"--optimizer adam --hutchinson_samples 100", "Samples = 100"),
        (f"--optimizer adam --hutchinson_samples 500", "Samples = 500"),
        (f"--optimizer adam --hutchinson_samples 1000", "Samples = 1000"),
        (f"--optimizer adam --hutchinson_samples 5000", "Samples = 5000"),
    ]
    cmds = [(f"{cmd} --steps {STEPS} --ckpts {CKPTS}", label) for cmd, label in cmds]
    save_fn = f"{ASSETS}/hutchinson_samples.png"
    main(cmds, steps=STEPS, read_fn=LOG_FN, save_fn=save_fn)