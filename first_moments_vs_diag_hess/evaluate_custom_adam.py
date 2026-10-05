from functools import partial

import numpy as np
from scipy.optimize import curve_fit
from torch.utils.data import Subset
import torch.nn as nn
from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from constants import *
from custom_adam import CustomAdam
from utilities import MLP, train_gd, loss_fn, compute_exp_avg_sq, compute_hess_diag, power_law_offset


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
    learn_scale: bool,
):

    full_trainset = Dataset(root=DATASTORE, train=True, download=True, transform=transforms.ToTensor())
    torch.manual_seed(seed)
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
    
    def fit(x, y):
        x, y = np.asarray(x), np.asarray(y)
        mask = (x > 1e-24) & (y > 1e-24)
        if sum(mask) >= 2:
            x, y = x[mask], y[mask]
            p0 = [0., 75., 0.5]
            bounds = (0, np.inf)
            (a, b, c), _ = curve_fit(power_law_offset, x, np.log(y), p0=p0, bounds=bounds, maxfev=10000)
            print(f"Updating power-law to y = {a:.2e} + {b:.2f} * (x ^ {c:.2f})")
            return (a, b, c)
        else:
            print("Too few values to fit power-law; defaulting to the rule y = 0.0 + x ^ 0.5")
            return (0.0, 1.0, 0.5)

    losses, tracked_vals = [], []
    tracker_fns = (compute_exp_avg_sq, partial(compute_hess_diag, hutchinson_samples=HUTCHINSON_SAMPLES))
    if learn_scale:
        exp_avg_sq, hess_diag = evaluate_tracker_fns(tracker_fns)
        tracked_vals.append([exp_avg_sq, hess_diag])
        optimizer.update_scale(fit(exp_avg_sq, hess_diag))

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
            if learn_scale:
                exp_avg_sq, hess_diag = evaluate_tracker_fns(tracker_fns)
                tracked_vals.append([exp_avg_sq, hess_diag])
                optimizer.update_scale(fit(exp_avg_sq, hess_diag))

    with torch.no_grad():
        losses.append(loss_fn(model(train_inputs), one_hot, loss_type).item())
    assets = (train_inputs, one_hot, model, optimizer)
    out = list(map(np.array, (losses,)))
    if learn_scale:
        out.extend(list(map(np.array, zip(*tracked_vals))))

    return assets, tuple(out)


if __name__ == "__main__":

    OUTPUT_DIM = 10
    MODEL, WIDTHS, MODEL_KWARGS = MLP, [64]*3, dict()
    ACTIVATION, ACT_KWARGS = "ReLU", dict()
    LOSS_TYPE = "Mean Squared Error"
    EPOCHS = 1000; LOG_EVERY = EPOCHS // 10
    save_fn = "evaluate-custom-adam.png"

    OPTIMIZER, OPTIM_KWARGS = CustomAdam, dict(lr=1e-3, betas=(0.9, 0.95))
    assets, out = train_gd(
        Dataset=datasets.MNIST, output_dim=OUTPUT_DIM, subset_size=SUBSET_SIZE,
        Model=MODEL, widths=WIDTHS, model_kwargs=MODEL_KWARGS,
        act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
        loss_type=LOSS_TYPE, Optimizer=OPTIMIZER, optim_kwargs=OPTIM_KWARGS,
        epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED, learn_scale=False,
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.5*1, 4.5*1))
    losses, *_ = out
    ax.plot(losses, linewidth=5, label="Original Adam")

    assets, out = train_gd(
        Dataset=datasets.MNIST, output_dim=OUTPUT_DIM, subset_size=SUBSET_SIZE,
        Model=MODEL, widths=WIDTHS, model_kwargs=MODEL_KWARGS,
        act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
        loss_type=LOSS_TYPE, Optimizer=OPTIMIZER, optim_kwargs=OPTIM_KWARGS,
        epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED, learn_scale=True,
    )

    losses, *_ = out
    ax.plot(losses, linewidth=5, label="Scale-adapted Adam")

    ax.grid()
    ax.legend()
    fig.tight_layout()
    if save_fn is not None:
        plt.savefig(save_fn)
    plt.close()