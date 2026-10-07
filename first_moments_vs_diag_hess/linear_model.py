"""
Comparing the GGN and Sq-Grads at the OLS solution.
Stupid -- grads are 0 at OLS solution...
"""


import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from constants import *
from utilities import compute_gradient, curve_fit, power_law_offset, power_law_offset_format
from check_ggn_vs_hess import compute_ggn_diag

LOSS_TYPE = "Mean Squared Error"
SAVE_DIR = "linear-model"


def loss_fn(outputs, one_hot, loss_type):
    if loss_type.lower() == "cross entropy loss":
        return F.cross_entropy(outputs, one_hot, reduction="sum") / outputs.size(0)
    elif loss_type.lower() == "mean squared error":
        return 0.5 * F.mse_loss(outputs, one_hot, reduction="sum") / outputs.size(0)

def loss_gradient(outputs, one_hot, loss_type):
    if loss_type.lower() == "cross entropy loss":
        return (torch.softmax(outputs, dim=1) - one_hot) / outputs.size(0)
    elif loss_type.lower() == "mean squared error":
        return (outputs - one_hot) / outputs.size(0)


Datasets = (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10)
output_dims = (10, 10, 10)
L = 1
d = 128

for Dataset, output_dim in zip(Datasets, output_dims):

    dataset_name = Dataset.__name__.lower()
    full_trainset = Dataset(
        root=f"{DATASTORE}/{dataset_name}",
        train=True, download=True, transform=transforms.ToTensor()
    )
    data_loader = torch.utils.data.DataLoader(full_trainset, batch_size=SUBSET_SIZE)
    x, y = next(iter(data_loader))

    # Note: we work with transposed coordinates, in comparison with the theory
    x = x.view(x.size(0), -1).float().to(DEVICE)    # (n, d_0)
    y = F.one_hot(y, num_classes=output_dim).float().to(DEVICE) # (n, d_L)
    w = torch.linalg.pinv(x) @ y  # (d_0, d_L)
    loss = loss_fn(x@w, y, loss_type="Mean Squared Error")
    print(f"OLS regression loss: {loss.item():.6e}")

    s_X = torch.linalg.svdvals(x)       # (d_0,)
    sigma = s_X / np.sqrt(x.size(0))
    u_1, s_W, v_L = torch.linalg.svd(w.to(DEVICE), full_matrices=False)   # (d_0, d_L), (d_L,), (d_L, d_L), assuming d_L < d_0
    u_1 = u_1
    s_W = s_W ** (1/L)    # Note: don't diag-embed here because torch computes 0^0 = 1
    v_L = v_L.T           # torch.linalg.svd returns the transposed right SVs
    
    weights = list()
    dims = [x.size(1)] + [d]*(L-1) + [output_dim]
    u = u_1.clone()
    for ell, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]), 1):
        v = torch.linalg.qr(torch.randn(d, output_dim, device=DEVICE))[0]
        w = (u*s_W) @ v.T
        weights.append(w)
        u = v
    weights[-1] = w @ v @ v_L.T

    model = list()
    with torch.no_grad():
        for w in weights:
            layer = nn.Linear(*w.size(), bias=False)
            layer.weight.copy_(w.T)
            model.append(layer)
    model = nn.Sequential(*model)
    model = model.to(DEVICE)
    with torch.no_grad():
        loss = loss_fn(model(x), y, loss_type=LOSS_TYPE)
        print(f"Simulated OLS solution's loss: {loss.item():.6e}")

    ggn_diag = compute_ggn_diag(
        x, y, model, loss_type=LOSS_TYPE, optimizer=None,
        device=DEVICE, hutchinson_samples=HUTCHINSON_SAMPLES
    )
    loss_fn(model(x), y, loss_type=LOSS_TYPE).backward()    # Populate the gradients
    gradient_sq = compute_gradient(
        x, y, model, loss_type=LOSS_TYPE, device=DEVICE,
        optimizer=torch.optim.SGD(model.parameters())
    ).square()

    fig, ax = plt.subplots(1, 1, figsize=(7.5*1, 4.5*1))
    x, y = np.asarray(ggn_diag.detach().cpu()), np.asarray(gradient_sq.detach().cpu())
    mask = (x > 1e-24) & (y > 1e-24)
    x, y = x[mask], y[mask]
    ax.scatter(x, y, s=1, color="yellowgreen", label="Data")
    p0 = [0., 75., 0.5]
    bounds = (0, np.inf)
    opt, _ = curve_fit(power_law_offset, x, np.log(y), p0=p0, bounds=bounds, maxfev=10000)
    x_fit = np.geomspace(x.min(), x.max(), 200)
    y_fit = np.exp(power_law_offset(x_fit, *opt))
    ax.plot(x_fit, y_fit, linewidth=5, linestyle="--", color="green", label=power_law_offset_format(*opt))
    ax.legend(fontsize=20, framealpha=1, markerscale=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_title(f"OLS Solution", fontsize=20)
    ax.set_xlabel("GGN Diagonal", fontsize=20)
    ax.set_ylabel("Squared Gradient", fontsize=20)

    fig.tight_layout()
    save_fn = f"linear-model/{dataset_name}_ols.png"
    if save_fn is not None:
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        plt.savefig(save_fn)
    plt.close()