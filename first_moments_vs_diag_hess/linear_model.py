from typing import Callable
import os

import numpy as np
from scipy.optimize import curve_fit
import torch
from torch import optim, nn
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from constants import DATASTORE, ASSETS
from utilities import Model, loss_fn, power_law_offset, power_law_offset_format
from main import as_list, parser, \
    dataset_map, optimizer_map, tracker_fns_map, make_optim_kwargs, make_tracker_fns


class MLP(Model):

    def __init__(self, widths: list[int], output_dim: int):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.LazyLinear(width, bias=False) for width in widths])
        self.readout = nn.LazyLinear(output_dim, bias=False)

    @torch.no_grad()
    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        self.forward(x[:1])     # Initialize lazy layers
        return x
    
    @torch.no_grad()
    def balanced_solution(self, W: torch.Tensor) -> None:
        # Setup
        all_layers = self._all_layers()
        L = len(all_layers)
        # Assert that the solution matches the layers' shapes, such that W = W_L W_{L-1} ... W_2 W_1
        assert W.ndim == 2
        assert W.size(0) == all_layers[-1].out_features
        assert W.size(1) == all_layers[0].in_features
        assert L == 1
        all_layers[0].weight.copy_(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for emb_layer in self.emb_layers:
            x = emb_layer(x)
        return self.readout(x)


def train_gd(
    Dataset,
    output_dim: int,
    subset_size: int,
    Model: nn.Module,               # incorrect type-hinting -- Model is not an instance of nn.Module
    widths: list[int],
    Optimizer: optim.Optimizer,     # incorrect type-hinting -- Optimizer is not an instance of optim.Optimizer
    optim_kwargs: dict,
    device: torch.device,
    seed: int,
    tracker_fns: list[Callable] = [],
):

    full_trainset = Dataset(
        root=f"{DATASTORE}/{Dataset.__name__.lower()}",
        train=True, download=True, transform=transforms.ToTensor()
    )
    torch.manual_seed(seed)
    indices = torch.randperm(len(full_trainset))[:subset_size]
    train_subset = Subset(full_trainset, indices)

    data_loader = DataLoader(train_subset, batch_size=subset_size)
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=output_dim).float()
    labels, one_hot = (labels==0).int(), one_hot[:, 0:1]    # One-v-all classification

    model = Model(widths=widths, output_dim=one_hot.size(1)).to(device)
    train_inputs = model.prepare_input(images)      # Also initializes lazy layers

    model.reset_parameters(seed=seed)
    model.train()
    optimizer = Optimizer(model.parameters(), **optim_kwargs)
    loss_type = "mean squared error"

    def evaluate_tracker_fns(tracker_fns):
        out = []
        for tracker_fn in tracker_fns:
            out.append(tracker_fn(
                train_inputs, one_hot, model, loss_type, optimizer, device
            ).detach().to("cpu"))
        return out

    def log_metrics(losses, tracked_vals):
        model.zero_grad()
        outputs = model(train_inputs)
        loss = loss_fn(outputs, one_hot, loss_type)
        loss.backward()
        losses.append(loss.item())
        tracked_vals.append(evaluate_tracker_fns(tracker_fns))
        predicted = torch.argmax(outputs.data, 1) if outputs.size(1) > 1 else (outputs.data > 0.5).int().squeeze(-1)
        acc = (predicted == labels).float().mean().item() * 100
        print(f'Loss: {loss.item():.4f}, Acc: {acc:.2f}%')
        return losses, tracked_vals

    losses, tracked_vals = [], []
    losses, tracked_vals = log_metrics(losses, tracked_vals)

    ols_soln = (torch.linalg.pinv(train_inputs) @ one_hot).T
    for _ in range(4):
        ols_soln_perturbed = ols_soln + 0.1 * ols_soln.abs().sqrt() * torch.randn_like(ols_soln)
        model.balanced_solution(ols_soln_perturbed)
        losses, tracked_vals = log_metrics(losses, tracked_vals)

    model.balanced_solution(ols_soln)
    losses, tracked_vals = log_metrics(losses, tracked_vals)

    assets = (train_inputs, one_hot, model, optimizer)
    out = list(map(np.array, (losses,)))
    out.extend(list(map(np.array, zip(*tracked_vals))))

    return assets, tuple(out)


def run(args):

    Dataset, output_dim = dataset_map[args.dataset]
    widths = as_list(args.widths)
    Optimizer = optimizer_map[args.optimizer]
    optim_kwargs = make_optim_kwargs(args)
    tracker_fns = make_tracker_fns(args)
    device = torch.device(args.device)

    assets, out = train_gd(
        Dataset=Dataset, output_dim=output_dim, subset_size=args.subset_size,
        Model=MLP, widths=widths, Optimizer=Optimizer, optim_kwargs=optim_kwargs,
        device=device, seed=args.seed, tracker_fns=tracker_fns,
    )

    return assets, out


def plot(xs, ys, ckpts, xlabel, ylabel, save_fn=None):

    fig, axs = plt.subplots(2, 3, figsize=(7.5*3, 4.5*2))
    for i, (ckpt, ax) in enumerate(zip(ckpts, axs.flatten())):
        x, y = np.asarray(xs[i]), np.asarray(ys[i])
        mask = (x > 1e-24) & (y > 1e-24)
        x, y = x[mask], y[mask]
        ax.scatter(x, y, s=6, color="yellowgreen", label="Data" if i==0 else None)
        if sum(mask) >= 2:
            p0 = [0., 75., 0.5]
            bounds = (0, np.inf)
            opt, _ = curve_fit(power_law_offset, x, np.log(y), p0=p0, bounds=bounds, maxfev=10000)
            x_fit = np.geomspace(x.min(), x.max(), 200)
            y_fit = np.exp(power_law_offset(x_fit, *opt))
            ax.plot(x_fit, y_fit, linewidth=5, linestyle="--", color="green", label=power_law_offset_format(*opt))
            ax.legend(fontsize=20, framealpha=1, markerscale=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.set_title(ckpt, fontsize=20)
    for ax in axs[-1,:]:
        ax.set_xlabel(xlabel, fontsize=20)
    for ax in axs[:,0]:
        ax.set_ylabel(ylabel, fontsize=20)

    fig.tight_layout()
    if save_fn is not None:
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        plt.savefig(save_fn)
    plt.close()


def driver(cmd: str, ckpts: list, save_fn: str) -> None:
    args = parser.parse_args(cmd.split())
    hess_idx = args.trackers.index("exact_hess_diag")
    args.bias = False
    _, out = run(args)
    tracked_vals = out[1:]
    labels = [tracker_fns_map[tracker][1] for tracker in args.trackers]
    ys, ylabel = tracked_vals[hess_idx], labels[hess_idx]
    for x_idx in range(len(args.trackers)):
        if x_idx == hess_idx:
            continue
        xs, xlabel = tracked_vals[x_idx], labels[x_idx]
        plot(xs, ys, ckpts, xlabel, ylabel, save_fn=save_fn.format(comparison=args.trackers[x_idx]))


driver(
    # NOTE: has to be static measures since we're not training the model
    "--optimizer adam --subset_size 1000 --widths [] --trackers gradient empirical_fisher_diag exact_hess_diag --seed 0",
    ["Random Initialization"] + ["Perturbed OLS Solution"]*4 + ["OLS Solution"],
    save_fn=ASSETS+"/theory/linear-{comparison}.png"
)