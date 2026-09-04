import warnings; warnings.filterwarnings("ignore")
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt

from constants import DATASTORE


def get_activation(act_name: str, **act_kwargs) -> nn.Module:

    if act_name is None:
        return nn.Identity()

    act_name = act_name.lower()
    if act_name == "identity" or act_name == "linear":
        return nn.Identity()
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "leakyrelu" or act_name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=act_kwargs.get("negative_slope", 0.01))
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()

    raise ValueError(f"Don't recognize `act_name={act_name}`.")

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def _all_layers(self) -> list[nn.Module]:
        return self.emb_layers + [self.readout]

    def reset_parameters(self, seed: int = None) -> None:
        torch.manual_seed(seed)
        for layer in self._all_layers():
            layer.reset_parameters()

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class MLP(Model):

    def __init__(
        self, widths: list[int], act_name: str, act_kwargs: dict, output_dim: int,
        normalization: bool = False,
    ):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.LazyLinear(width, bias=False) for width in widths])
        self.activation = get_activation(act_name, **act_kwargs)
        self.normalization = nn.LayerNorm(widths[-1], eps=1e-12, elementwise_affine=False) if normalization else nn.Identity()
        self.readout = nn.LazyLinear(output_dim, bias=False)

    @torch.no_grad()
    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        self.forward(x[:1])     # Initialize lazy layers
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for emb_layer in self.emb_layers:
            x = self.activation(emb_layer(x))
        x = self.normalization(x)
        return self.readout(x)

def loss_fn(outputs, one_hot, loss_type):
    if loss_type.lower() == "cross entropy loss":
        return F.cross_entropy(outputs, one_hot, reduction="sum") / outputs.size(0)
    elif loss_type.lower() == "mean squared error":
        return 0.5 * F.mse_loss(outputs, one_hot, reduction="sum") / outputs.size(0)
    
def train_gd(
    Dataset,
    output_dim: int,
    subset_size: int,
    Model: nn.Module,
    widths: list[int],
    model_kwargs: dict,
    act_name: str,
    act_kwargs: dict,
    loss_type: str,
    Optimizer: optim.Optimizer,
    optim_kwargs: dict,
    epochs: int,
    log_every: int,
    device: torch.device,
    seed: int,
    tracker_fns: list[Callable] = [],
):

    full_trainset = Dataset(root=DATASTORE, train=True, download=True, transform=transforms.ToTensor())
    indices = torch.randperm(len(full_trainset))[:subset_size]
    train_subset = Subset(full_trainset, indices)

    data_loader = torch.utils.data.DataLoader(train_subset, batch_size=subset_size)
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=output_dim).float()

    model = Model(
        widths=widths, output_dim=output_dim,
        act_name=act_name, act_kwargs=act_kwargs,
        **model_kwargs,
    ).to(device)

    train_inputs = model.prepare_input(images)      # Also initializes lazy layers
    model.reset_parameters(seed=seed)

    optimizer = Optimizer(model.parameters(), **optim_kwargs)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100, threshold=1e-3, threshold_mode="rel")

    def evaluate_tracker_fns(tracker_fns):
        out = []
        for tracker_fn in tracker_fns:
            out.append(tracker_fn(train_inputs, one_hot, model, loss_type, optimizer, device).detach().to("cpu"))
        return out

    losses, tracked_vals = [], []
    tracked_vals.append(evaluate_tracker_fns(tracker_fns))

    for epoch in range(1, epochs+1):

        model.train()
        optimizer.zero_grad()
        outputs = model(train_inputs)
        loss = loss_fn(outputs, one_hot, loss_type)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach().item())

        losses.append(loss.detach().item())
        if epoch % log_every == 0:
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).float().mean().item() * 100
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.2e}, Acc: {acc:.2f}%')
        if epoch % log_every == 0:
            tracked_vals.append(evaluate_tracker_fns(tracker_fns))

    with torch.no_grad():
        losses.append(loss_fn(model(train_inputs), one_hot, loss_type).item())
    assets = (train_inputs, one_hot, model)
    out = list(map(np.array, (losses,)))
    out.extend(list(map(np.array, zip(*tracked_vals))))

    return assets, tuple(out)

@torch.no_grad()
def compute_gradient(X, Y, model, loss_type, optimizer, device):
    gradient = [
        (param.grad if param.grad is not None else torch.zeros_like(param)).flatten()
        for group in optimizer.param_groups
        for param in group["params"]
        if param.requires_grad
    ]
    return torch.hstack(gradient).abs()

@torch.no_grad()
def compute_momentum(X, Y, model, loss_type, optimizer, device):
    momentum = [
        optimizer.state[param].get("momentum_buffer", torch.zeros_like(param.data)).flatten()
        for group in optimizer.param_groups
        for param in group["params"]
        if param.requires_grad
    ]
    return torch.hstack(momentum).abs()

@torch.no_grad()
def compute_exp_avg(X, Y, model, loss_type, optimizer, device):
    exp_avg_sq = [
        optimizer.state[param].get("exp_avg", torch.zeros_like(param.data)).flatten() / \
            (1 - group["betas"][0] / (optimizer.state[param].get("step", 0) + 1e-12))       # Can debias but it barely changes anything
        for group in optimizer.param_groups
        for param in group["params"]
        if param.requires_grad
    ]
    return torch.hstack(exp_avg_sq).abs()

@torch.no_grad()
def compute_exp_avg_sq(X, Y, model, loss_type, optimizer, device):
    exp_avg_sq = [
        optimizer.state[param].get("exp_avg_sq", torch.zeros_like(param.data)).flatten() / \
            (1 - group["betas"][1] / (optimizer.state[param].get("step", 0) + 1e-12))       # Can debias but it barely changes anything
        for group in optimizer.param_groups
        for param in group["params"]
        if param.requires_grad
    ]
    return torch.hstack(exp_avg_sq)

def compute_hvp(X, Y, model, vector, loss_type, device):
    model.zero_grad()
    vector = vector.to(device)
    loss = loss_fn(model(X), Y, loss_type)
    grad = autograd.grad(loss, model.parameters(), create_graph=True)
    dot = parameters_to_vector(grad).mul(vector).sum()
    hvp = autograd.grad(dot, model.parameters())
    hvp = parameters_to_vector([v.contiguous() for v in hvp])
    return hvp

def compute_hess_diag(X, Y, model, loss_type, optimizer, device, hutchinson_samples):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sum_samples = 0.
    for i in range(1, hutchinson_samples+1):
        random_vector = torch.randint(2, (num_params,), device=device, dtype=X.dtype).mul(2).sub(1)
        sum_samples += random_vector * compute_hvp(X, Y, model, random_vector, loss_type, device)
    hess_diag = sum_samples / hutchinson_samples
    return hess_diag

def power_law_offset(x, a, b, c):
    return np.log(a + b * (x ** c))

def plot(xs, ys, ckpts, log_every, xlabel, ylabel, save_fn=None):

    fig, axs = plt.subplots(2, 3, figsize=(7.5*3, 4.5*2))
    for i, (ckpt, ax) in enumerate(zip(ckpts, axs.flatten())):
        x, y = np.asarray(xs[ckpt]), np.asarray(ys[ckpt])
        mask = (x > 1e-24) & (y > 1e-16)
        x, y = x[mask], y[mask]
        ax.scatter(x, y, s=1, color="green", alpha=0.5, label="Data" if i==0 else None)
        if sum(mask) >= 2:
            p0 = [0., 75., 0.5]
            bounds = (0, np.inf)
            try:
                (a_fit, b_fit, c_fit), _ = curve_fit(power_law_offset, x, np.log(y), p0=p0, bounds=bounds, maxfev=10000)
                x_fit = np.geomspace(x.min(), x.max(), 200)
                y_fit = np.exp(power_law_offset(x_fit, a_fit, b_fit, c_fit))
                mantissa, exponent = f"{a_fit:.2e}".split("e")
                ax.plot(x_fit, y_fit, color="red", linestyle="--", linewidth=3, label=f"Fit: $y = {mantissa} \\cdot 10^{{{exponent}}} + {b_fit:.2f} \\cdot x^{{{c_fit:.2f}}}$")
                ax.legend(fontsize=16, framealpha=1, markerscale=6)
            except RuntimeError:
                pass
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_title(f"Checkpoint {ckpt*log_every}", fontsize=16)
    for ax in axs[-1,:]:
        ax.set_xlabel(xlabel, fontsize=16)
    for ax in axs[:,0]:
        ax.set_ylabel(ylabel, fontsize=16)

    fig.tight_layout()
    if save_fn is not None:
        plt.savefig(save_fn)