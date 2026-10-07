from constants import ASSETS
from main import parser, tracker_fns_map, run
from utilities import plot


def driver(cmd: str, ckpts: list, save_fn: str) -> None:
    args = parser.parse_args(cmd.split())
    _, out = run(args)
    tracked_vals = out[1:]
    labels = [tracker_fns_map[tracker][1] for tracker in args.trackers]
    log_every = args.epochs // args.ckpts
    plot(*tracked_vals, ckpts, *labels, log_every, save_fn=save_fn)


if __name__ == "__main__":

    cmds = [
        ("--optimizer sgd --beta1 0.0 --epochs 10000 --trackers gradient hess_diag",    f"{ASSETS}/sgd/gradient.png"),
        ("--optimizer sgd --epochs 10000 --trackers momentum hess_diag",                f"{ASSETS}/sgd/momentum.png"),
        ("--optimizer adam",                                                            f"{ASSETS}/adam/baseline.png"),
        ("--optimizer adam --subset_size 25000 --epochs 2000",                          f"{ASSETS}/adam/25k.png"),
        ("--optimizer adam --loss_type xent --epochs 2000",                             f"{ASSETS}/adam/xent.png"),
        ("--optimizer adam --dataset fmnist --epochs 2000",                             f"{ASSETS}/adam/fmnist.png"),
        ("--optimizer adam --dataset cifar10 --epochs 2000",                            f"{ASSETS}/adam/cifar10.png"),
        ("--optimizer adam --dataset cifar100 --epochs 5000",                           f"{ASSETS}/adam/cifar100.png"),
        ("--optimizer adam --widths [256]*3 --epochs 500",                              f"{ASSETS}/adam/wide.png"),
        ("--optimizer adam --widths [64]*7",                                            f"{ASSETS}/adam/deep.png"),
        ("--optimizer adam --activation tanh",                                          f"{ASSETS}/adam/tanh.png"),
        ("--optimizer adam --beta2 0.9",                                                f"{ASSETS}/adam/beta2=0.9.png"),
        ("--optimizer adam --beta2 0.99",                                               f"{ASSETS}/adam/beta2=0.99.png"),
        ("--optimizer adam --eps 1e-6 --epochs 2000",                                   f"{ASSETS}/adam/eps=1e-6.png"),
        ("--optimizer adamw",                                                           f"{ASSETS}/adamw/baseline.png"),
        ("--optimizer adamw --subset_size 25000 --epochs 2000",                         f"{ASSETS}/adamw/25k.png"),
        ("--optimizer adamw --loss_type xent --epochs 2000",                            f"{ASSETS}/adamw/xent.png"),
        ("--optimizer adamw --dataset fmnist --epochs 2000",                            f"{ASSETS}/adamw/fmnist.png"),
        ("--optimizer adamw --dataset cifar10 --epochs 2000",                           f"{ASSETS}/adamw/cifar10.png"),
        ("--optimizer adamw --dataset cifar100 --epochs 5000",                          f"{ASSETS}/adamw/cifar100.png"),
        ("--optimizer adamw --widths [256]*3 --epochs 500",                             f"{ASSETS}/adamw/wide.png"),
        ("--optimizer adamw --widths [64]*7",                                           f"{ASSETS}/adamw/deep.png"),
        ("--optimizer adamw --activation tanh",                                         f"{ASSETS}/adamw/tanh.png"),
        ("--optimizer adamw --beta2 0.9",                                               f"{ASSETS}/adamw/beta2=0.9.png"),
        ("--optimizer adamw --beta2 0.99",                                              f"{ASSETS}/adamw/beta2=0.99.png"),
        ("--optimizer adamw --eps 1e-6 --epochs 2000",                                  f"{ASSETS}/adamw/eps=1e-6.png"),
    ]

    ckpts = [1, 3, 5, 7, 9, 10]

    for cmd, save_fn in cmds:
        print(f"*** python main.py {cmd} ***")
        driver(cmd, ckpts, save_fn)
        print()