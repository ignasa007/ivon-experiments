from constants import ASSETS
from main import parser, tracker_fns_map, run
from utilities import plot


def driver(cmd: str, ckpts: list, save_fn: str) -> None:
    args = parser.parse_args(cmd.split())
    _, out = run(args)
    tracked_vals = out[1:]
    labels = [tracker_fns_map[tracker][1] for tracker in args.trackers]
    log_every = args.steps // args.ckpts
    plot(*tracked_vals, ckpts, *labels, log_every, save_fn=save_fn)


if __name__ == "__main__":

    cmds = [
        ("--optimizer sgd --beta1 0.0 --steps 10000 --trackers gradient hess_diag",     f"{ASSETS}/sgd/gradient.png"),
        ("--optimizer sgd --steps 10000 --trackers momentum hess_diag",                 f"{ASSETS}/sgd/momentum.png"),
        ("--optimizer adam --beta1 0.0 --lr 1e-1 --eps 100 --steps 10000",              f"{ASSETS}/sgd/expavgsq.png"),
    ]
    for optimizer in ("adam", "adamw"):
        cmds += [
            (f"--optimizer {optimizer}",                                                f"{ASSETS}/{optimizer}/baseline.png"),
            (f"--optimizer {optimizer} --subset_size 25000 --steps 2000",               f"{ASSETS}/{optimizer}/25k.png"),
            (f"--optimizer {optimizer} --loss_type xent --steps 2000",                  f"{ASSETS}/{optimizer}/xent.png"),
            (f"--optimizer {optimizer} --dataset fmnist --steps 2000",                  f"{ASSETS}/{optimizer}/fmnist.png"),
            (f"--optimizer {optimizer} --dataset cifar10 --steps 2000",                 f"{ASSETS}/{optimizer}/cifar10.png"),
            (f"--optimizer {optimizer} --dataset cifar100 --steps 5000",                f"{ASSETS}/{optimizer}/cifar100.png"),
            (f"--optimizer {optimizer} --widths [256]*3 --steps 500",                   f"{ASSETS}/{optimizer}/wide.png"),
            (f"--optimizer {optimizer} --widths [64]*7",                                f"{ASSETS}/{optimizer}/deep.png"),
            (f"--optimizer {optimizer} --activation tanh",                              f"{ASSETS}/{optimizer}/tanh.png"),
            (f"--optimizer {optimizer} --beta2 0.9",                                    f"{ASSETS}/{optimizer}/beta2=0.9.png"),
            (f"--optimizer {optimizer} --beta2 0.99",                                   f"{ASSETS}/{optimizer}/beta2=0.99.png"),
            (f"--optimizer {optimizer} --eps 1e-6 --steps 2000",                        f"{ASSETS}/{optimizer}/eps=1e-6.png"),
        ]

    ckpts = [1, 3, 5, 7, 9, 10]
    for cmd, save_fn in cmds:
        print(f"*** python main.py {cmd} ***")
        driver(cmd, ckpts, save_fn)
        print()