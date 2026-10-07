from argparse import ArgumentParser
from functools import partial

from torchvision import datasets
from torch import optim

from constants import *
from utilities import MLP, train_gd, plot, \
    compute_gradient, compute_momentum, compute_exp_avg, compute_exp_avg_sq, \
    exact_hess_diag, estimate_hess_diag, estimate_ggn_diag


def as_list(s):
    lsts = [eval(ss) for ss in s.split()]
    assert all(isinstance(lst, list) for lst in lsts)
    return sum(lsts, start=list())

dataset_map = {
    "mnist": (datasets.MNIST, 10),
    "fmnist": (datasets.FashionMNIST, 10),
    "cifar10": (datasets.CIFAR10, 10),
    "cifar100": (datasets.CIFAR100, 100),
}

model_map = {
    "mlp": MLP,
}

loss_type_map = {
    "mse": "mean squared error",
    "xent": "cross entropy loss",
}

optimizer_map = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
}

tracker_fns_map = {
    "gradient": (compute_gradient, "Abs Gradient"),
    "momentum": (compute_momentum, "Abs Momentum"),
    "exp_avg": (compute_exp_avg, "Exp Avg"),
    "exp_avg_sq": (compute_exp_avg_sq, "Exp Avg Sq"),
    "exact_hess_diag": (exact_hess_diag, "Exact Hess Diag"),
    "hess_diag": (estimate_hess_diag, "Hessian Diagonal"),
    "ggn_diag": (estimate_ggn_diag, "GGN Diagonal"),
}

def make_optim_kwargs(args):
    if args.optimizer == "sgd":
        return dict(lr=args.lr, momentum=args.beta1)
    elif args.optimizer in ("adam", "adamw"):
        return dict(lr=args.lr, betas=(args.beta1, args.beta2))
    
def make_tracker_fns(args):
    tracker_fns = list()
    for tracker in args.trackers:
        tracker_fn, _ = tracker_fns_map[tracker]
        if tracker in ("gnn_diag", "hess_diag"):
            tracker_fn = partial(tracker_fn, hutchinson_samples=args.hutchinson_samples)
        tracker_fns.append(tracker_fn)
    return tracker_fns


parser = ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str.lower,
    default="mnist",
    choices=dataset_map.keys()
)
parser.add_argument(
    "--subset_size",
    type=int,
    default=SUBSET_SIZE
)
parser.add_argument(
    "--model",
    type=str.lower,
    default="mlp",
    choices=model_map.keys()  
)
parser.add_argument(
    "--widths",
    type=str,
    default="[64]*3"
)
parser.add_argument(
    "--bias",
    type=bool,
    default=True
)
parser.add_argument(
    "--normalization",
    type=bool,
    default=False
)
parser.add_argument(
    "--activation",
    type=str.lower,
    default="relu",
    choices=["identity", "linear", "relu", "tanh", "sigmoid"]
)
parser.add_argument(
    "--loss_type",
    type=str.lower,
    default="mse",
    choices=["mse", "xent"]
)
parser.add_argument(
    "--optimizer",
    type=str.lower,
    default="adam",
    choices=optimizer_map.keys() 
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3
)
parser.add_argument(
    "--beta1",
    type=float,
    default=0.9
)
parser.add_argument(
    "--beta2",
    type=float,
    default=0.95
)
parser.add_argument(
    "--eps",
    type=float,
    default=1e-8
)
parser.add_argument(
    "--epochs",
    type=int,
    default=1000
)
parser.add_argument(
    "--ckpts",
    type=int,
    default=CKPTS
)
parser.add_argument(
    "--trackers",
    nargs="*",
    type=str,
    default=["exp_avg_sq", "hess_diag"],
    choices=tracker_fns_map.keys()
)
parser.add_argument(
    "--hutchinson_samples",
    type=int,
    default=HUTCHINSON_SAMPLES
)
parser.add_argument(
    "--device",
    type=str,
    default=DEVICE,
    choices=["cpu", "cuda"]
)
parser.add_argument(
    "--seed",
    type=int,
    default=SEED
)


def run(args):

    Dataset, output_dim = dataset_map[args.dataset]
    Model = model_map[args.model]
    widths = as_list(args.widths)
    model_kwargs = dict(bias=args.bias, normalization=args.normalization)
    loss_type = loss_type_map[args.loss_type]
    Optimizer = optimizer_map[args.optimizer]
    optim_kwargs = make_optim_kwargs(args)
    tracker_fns = make_tracker_fns(args)
    log_every = args.epochs // args.ckpts
    device = torch.device(args.device)

    assets, out = train_gd(
        Dataset=Dataset, subset_size=args.subset_size, output_dim=output_dim,
        Model=Model, widths=widths, model_kwargs=model_kwargs,
        act_name=args.activation, act_kwargs=dict(),    # don't currently support `act_kwargs`
        loss_type=loss_type, Optimizer=Optimizer, optim_kwargs=optim_kwargs,
        epochs=args.epochs, log_every=log_every,
        device=device, seed=args.seed, tracker_fns=tracker_fns,
    )

    return assets, out


if __name__ == "__main__":

    args = parser.parse_args()
    assets, out = run(args)

    optimizer, tracked_vals = assets[-1], out[1:]
    ckpts = [1, 3, 5, 7, 9, 10]
    labels = [tracker_fns_map[tracker][1] for tracker in args.trackers]
    log_every = args.epochs // args.ckpts
    save_fn = f"{optimizer.__name__.lower()}/{args.save_fn}.png"
    plot(*tracked_vals, ckpts, *labels, log_every, save_fn=save_fn)