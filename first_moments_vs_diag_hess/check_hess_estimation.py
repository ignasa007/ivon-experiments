"""
Verify the implementation by comparing the analytically computed Hessian-diagonal against the MC-estimated one.
"""


from functools import partial

import numpy as np
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt

from constants import *
from utilities import MLP, compute_hess_diag, loss_fn, train_gd


OUTPUT_DIM = 10
MODEL, WIDTHS, MODEL_KWARGS = MLP, [32]*1, dict()
ACTIVATION, ACT_KWARGS = "ReLU", dict()
LOSS_TYPE = "Mean Squared Error"
OPTIMIZER, OPTIM_KWARGS = optim.Adam, dict(lr=1e-3, betas=(0.9, 0.95))
EPOCHS = 1000; LOG_EVERY = EPOCHS // CKPTS


def analytical_hess_diag(X, Y, model, loss_type, optimizer, device):
    # Follows https://github.com/HeyShinde/torch-secorder/blob/main/torch_secorder/core/hessian_diagonal.py
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss_fn(model(X), Y, loss_type=loss_type), params, create_graph=True)
    hess_diag = list()
    for param, grad in zip(params, grads):
        grad = grad.flatten()
        block_diag = torch.zeros_like(grad)
        for idx in range(grad.numel()):
            grad2 = torch.autograd.grad(grad[idx], param, retain_graph=True)[0]
            block_diag[idx] = grad2.flatten()[idx]
        hess_diag.append(block_diag.detach())
    return torch.hstack(hess_diag).cpu()


assets, out = train_gd(
    Dataset=torchvision.datasets.MNIST, output_dim=OUTPUT_DIM, subset_size=SUBSET_SIZE,
    Model=MODEL, widths=WIDTHS, model_kwargs=MODEL_KWARGS,
    act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
    loss_type=LOSS_TYPE, Optimizer=OPTIMIZER, optim_kwargs=OPTIM_KWARGS,
    epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED,
    tracker_fns=(analytical_hess_diag, partial(compute_hess_diag, hutchinson_samples=10_000))
)

losses, analytical_hess_diags, numerical_hess_diags = out
ckpts = [1, 3, 5, 7, 9, 10]
xlabel = "Analytical Hess Diag"
ylabel = "Numerical Hess Diag"

fig, axs = plt.subplots(2, 3, figsize=(7.5*3, 4.5*2))
for i, (ckpt, ax) in enumerate(zip(ckpts, axs.flatten())):
    x, y = np.asarray(analytical_hess_diags[ckpt]), np.asarray(numerical_hess_diags[ckpt])
    ax.scatter(x, y, s=1, color="yellowgreen", label="Data" if i==0 else None)
    identity = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), num=2)
    ax.plot(identity, identity, linewidth=5, linestyle="--", color="green", label="Identity" if i==0 else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    ax.legend(fontsize=20, framealpha=1, markerscale=8)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_title(f"Checkpoint {ckpt*LOG_EVERY}", fontsize=20)
for ax in axs[-1,:]:
    ax.set_xlabel(xlabel, fontsize=20)
for ax in axs[:,0]:
    ax.set_ylabel(ylabel, fontsize=20)

fig.tight_layout()
save_fn = f"{OPTIMIZER.__name__.lower()}/check-hess-estimation.png"
if save_fn is not None:
    plt.savefig(save_fn)
plt.close()