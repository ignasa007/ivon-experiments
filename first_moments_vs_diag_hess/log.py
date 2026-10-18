import os
from argparse import ArgumentParser
import string
from ast import literal_eval
import pickle
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

from main import parser, run
from utilities import power_law_offset


RESULTS_DIR = "results"


def main(cmd: str, write_fn: str, redo: bool = False, overwrite: bool = False) -> None:

    alphabet = list(string.ascii_letters + string.digits)
    run_id = "".join(np.random.choice(alphabet, 12))
    output_fn = f"{run_id}.pickle"
    
    args = parser.parse_args(cmd.split())
    row = vars(args).copy()

    if not redo and os.path.exists(write_fn):
        df = pd.read_csv(write_fn, index_col="timestamp")
        df.trackers = df.trackers.apply(literal_eval)
        if all(col in df.columns for col in row.keys()):
            matches = (
                df[list(row.keys())].eq(pd.Series(row))
                | (df[list(row.keys())].isna() & pd.Series(row).isna())
            ).all(axis=1)
            if matches.any():
                print(f"Experiment already logged and `redo={redo}`; skipping run.")
                return

    _, (losses, *tracked_vals) = run(args)
    log_every = args.steps // args.ckpts
    laws = list()
    for i, (x, y) in enumerate(zip(*tracked_vals)):
        mask = (x > 1e-24) & (y > 1e-24)
        if sum(mask) <= 2:
            continue
        x, y = x[mask], y[mask]
        p0 = [0., 75., 0.5]
        bounds = (0, np.inf)
        opt, _ = curve_fit(power_law_offset, x, np.log(y), p0=p0, bounds=bounds, maxfev=10000)
        laws.append((i*log_every, *opt))

    with open(f"{RESULTS_DIR}/{output_fn}", "wb") as f:
        pickle.dump({"args": args, "losses": losses, "laws": laws}, f, protocol=pickle.HIGHEST_PROTOCOL)

    if overwrite:
        df = pd.read_csv(write_fn, index_col="timestamp")
        df.trackers = df.trackers.apply(literal_eval)
        if all(col in df.columns for col in row.keys()):
            matches = (
                df[list(row.keys())].eq(pd.Series(row))
                | (df[list(row.keys())].isna() & pd.Series(row).isna())
            ).all(axis=1)
            if matches.any():
                df = df.loc[~matches]
                df.to_csv(write_fn, header=True)
                print("Removed previous matching configurations.")

    row["output_fn"] = output_fn
    timestamp = datetime.now(timezone.utc).isoformat()
    df = pd.DataFrame([row], index=[timestamp])
    df.index.name = "timestamp"
    os.makedirs(os.path.dirname(write_fn), exist_ok=True)
    df.to_csv(write_fn, mode="a", header=not os.path.exists(write_fn) or os.stat(write_fn).st_size==0)


if __name__ == "__main__":

    log_parser = ArgumentParser()
    log_parser.add_argument("--redo", action="store_true")
    log_parser.add_argument("--overwrite", action="store_true")
    args = log_parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    LOG_FN = f"{RESULTS_DIR}/logs.csv"

    # Datasets
    STEPS, CKPTS = 5000, 20
    cmds = [
        f"--optimizer adam --dataset mnist",
        f"--optimizer adam --dataset fmnist",
        f"--optimizer adam --dataset cifar10",
        f"--optimizer adam --dataset cifar100",
    ]
    for cmd in cmds:
        print(f"*** python main.py {cmd} --steps {STEPS} --ckpts {CKPTS} ***")
        main(
            f"{cmd} --steps {STEPS} --ckpts {CKPTS}",
            write_fn=LOG_FN, redo=args.redo, overwrite=args.overwrite
        )
        print()

    # Subset Sizes
    STEPS, CKPTS = 2000, 20
    cmds = [
        f"--optimizer adam --subset_size 5000",
        f"--optimizer adam --subset_size 10000",
        f"--optimizer adam --subset_size 15000",
        f"--optimizer adam --subset_size 20000",
        f"--optimizer adam --subset_size 25000",
    ]
    for cmd in cmds:
        print(f"*** python main.py {cmd} --steps {STEPS} --ckpts {CKPTS} ***")
        main(
            f"{cmd} --steps {STEPS} --ckpts {CKPTS}",
            write_fn=LOG_FN, redo=args.redo, overwrite=args.overwrite
        )
        print()

    # Widths
    STEPS, CKPTS = 1000, 20
    cmds = [
        f"--optimizer adam --widths [64]*3",
        f"--optimizer adam --widths [128]*3",
        f"--optimizer adam --widths [256]*3",
        f"--optimizer adam --widths [512]*3",
    ]
    for cmd in cmds:
        print(f"*** python main.py {cmd} --steps {STEPS} --ckpts {CKPTS} ***")
        main(
            f"{cmd} --steps {STEPS} --ckpts {CKPTS}",
            write_fn=LOG_FN, redo=args.redo, overwrite=args.overwrite
        )
        print()

    # Depths
    STEPS, CKPTS = 1000, 20
    cmds = [
        f"--optimizer adam --widths [64]*1",
        f"--optimizer adam --widths [64]*3",
        f"--optimizer adam --widths [64]*5",
        f"--optimizer adam --widths [64]*7",
    ]
    for cmd in cmds:
        print(f"*** python main.py {cmd} --steps {STEPS} --ckpts {CKPTS} ***")
        main(
            f"{cmd} --steps {STEPS} --ckpts {CKPTS}",
            write_fn=LOG_FN, redo=args.redo, overwrite=args.overwrite
        )
        print()

    # Activations
    STEPS, CKPTS = 1000, 20
    cmds = [
        f"--optimizer adam --activation linear",
        f"--optimizer adam --activation relu",
        f"--optimizer adam --activation leakyrelu",
        f"--optimizer adam --activation tanh",
        f"--optimizer adam --activation sigmoid",
    ]
    for cmd in cmds:
        print(f"*** python main.py {cmd} --steps {STEPS} --ckpts {CKPTS} ***")
        main(
            f"{cmd} --steps {STEPS} --ckpts {CKPTS}",
            write_fn=LOG_FN, redo=args.redo, overwrite=args.overwrite
        )
        print()

    # Beta1
    STEPS, CKPTS = 2000, 20
    cmds = [
        f"--optimizer adam --beta1 0.8",
        f"--optimizer adam --beta1 0.9",
        f"--optimizer adam --beta1 0.95",
        f"--optimizer adam --beta1 0.99",
        f"--optimizer adam --beta1 0.999",
    ]
    for cmd in cmds:
        print(f"*** python main.py {cmd} --steps {STEPS} --ckpts {CKPTS} ***")
        main(
            f"{cmd} --steps {STEPS} --ckpts {CKPTS}",
            write_fn=LOG_FN, redo=args.redo, overwrite=args.overwrite
        )
        print()

    # Beta2
    STEPS, CKPTS = 2000, 20
    cmds = [
        f"--optimizer adam --beta2 0.8",
        f"--optimizer adam --beta2 0.9",
        f"--optimizer adam --beta2 0.95",
        f"--optimizer adam --beta2 0.99",
        f"--optimizer adam --beta2 0.999",
    ]
    for cmd in cmds:
        print(f"*** python main.py {cmd} --steps {STEPS} --ckpts {CKPTS} ***")
        main(
            f"{cmd} --steps {STEPS} --ckpts {CKPTS}",
            write_fn=LOG_FN, redo=args.redo, overwrite=args.overwrite
        )
        print()

    # Epsilon
    STEPS, CKPTS = 2000, 20
    cmds = [
        f"--optimizer adam --eps 1e-8",
        f"--optimizer adam --eps 1e-7",
        f"--optimizer adam --eps 1e-6",
        f"--optimizer adam --eps 1e-5",
    ]
    for cmd in cmds:
        print(f"*** python main.py {cmd} --steps {STEPS} --ckpts {CKPTS} ***")
        main(
            f"{cmd} --steps {STEPS} --ckpts {CKPTS}",
            write_fn=LOG_FN, redo=args.redo, overwrite=args.overwrite
        )
        print()

    # Weight Decay
    STEPS, CKPTS = 2000, 20
    cmds = [
        f"--optimizer adam --weight_decay 0",
        f"--optimizer adam --weight_decay 1e-4",
        f"--optimizer adam --weight_decay 1e-3",
        f"--optimizer adam --weight_decay 1e-2",
    ]
    for cmd in cmds:
        print(f"*** python main.py {cmd} --steps {STEPS} --ckpts {CKPTS} ***")
        main(
            f"{cmd} --steps {STEPS} --ckpts {CKPTS}",
            write_fn=LOG_FN, redo=args.redo, overwrite=args.overwrite
        )
        print()

    # Batch Size
    STEPS, CKPTS = 10000, 20
    cmds = [
        f"--optimizer adam --batch_size 100",
        f"--optimizer adam --batch_size 500",
        f"--optimizer adam --batch_size 1000",
        f"--optimizer adam --batch_size 5000",
    ]
    for cmd in cmds:
        print(f"*** python main.py {cmd} --steps {STEPS} --ckpts {CKPTS} ***")
        main(
            f"{cmd} --steps {STEPS} --ckpts {CKPTS}",
            write_fn=LOG_FN, redo=args.redo, overwrite=args.overwrite
        )
        print()