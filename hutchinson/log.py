'''
python -m hutchinson.log
'''


import os
import math
import re

from tqdm import tqdm
import torch
from torch import autograd
from torch.nn.utils import parameters_to_vector

from common.trainutils import loadcheckpoint
from common.dataloaders import TRAINDATALOADERS, CIFAR10Info


def compute_hvp(X, Y, model, vector):
    model.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(X), Y)
    grad = autograd.grad(loss, model.parameters(), create_graph=True)
    dot = parameters_to_vector(grad).mul(vector).sum()
    hvp = autograd.grad(dot, model.parameters())
    hvp = parameters_to_vector([v.contiguous() for v in hvp])
    return hvp

def compute_hess_diag(train_loader, model, data_samples, hutchinson_samples):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if data_samples > len(train_loader.dataset):
        print(
            f"data_samples={data_samples} > len(dataset)={len(train_loader.dataset)}; "
            f"using data_samples={len(train_loader.dataset)}"
        )
        data_samples = len(train_loader.dataset)
    remaining_samples = data_samples
    sum_samples = 0.
    for X, Y in tqdm(train_loader, total=math.ceil(round(data_samples/train_loader.batch_size, 8))):
        X, Y = map(lambda t: t[:remaining_samples].to(torch.device("cuda"), non_blocking=True), (X, Y))
        batch_weight = len(X) / data_samples
        for _ in range(hutchinson_samples):
            # rand_vector = torch.randn(num_params, device="cuda")
            # Rademacher variables have a lower variance than normal variables,
            # allowing the use of smaller `hutchinson_samples`
            rand_vector = torch.randint(2, size=(num_params,), device="cuda", dtype=X.dtype).mul(2).sub(1)
            sum_samples += batch_weight * rand_vector * compute_hvp(X, Y, model, rand_vector)
        remaining_samples -= len(X)
        if remaining_samples <= 0:
            break
    hess_diag = sum_samples / hutchinson_samples
    return hess_diag

def main(exp_dir, data_samples, hutchinson_samples, overwrite=False):

    save_dir = f"{exp_dir}/hutchinson/ds={data_samples}_hs={hutchinson_samples}"
    os.makedirs(save_dir, exist_ok=True)

    # Load the entire dataset as a validation set, so that there's no random augmentations
    # NOTE: there's no randomness in sampling the validation set (idk why; I think oki to remove)
    _, train_loader = TRAINDATALOADERS["cifar10"](
        data_dir="./datasets", train_val_split=1-data_samples/CIFAR10Info.counts["train"],
        # Batch size is memory-bound
        workers=1, pin_memory=True, tbatch=50, vbatch=2500
    )

    matches_left = 6
    for fn in sorted(os.listdir(exp_dir))[::-1]:
        if matches_left == 0:
            break
        match = re.search(r"checkpoint(\d+)\.pt", fn)
        if not match:
            continue
        matches_left -= 1
        save_fn = f"{save_dir}/{fn}"
        if os.path.exists(save_fn) and not overwrite:
            print(f"{save_fn} exists and overwrite={overwrite}; skipping")
            continue
        _, model, _, _, _ = loadcheckpoint(f"{exp_dir}/{fn}", device="cuda")
        model.eval()
        hess_diag = compute_hess_diag(train_loader, model, data_samples, hutchinson_samples)
        torch.save(hess_diag.to("cpu"), f=save_fn)


if __name__ == "__main__":

    DATA_SAMPLES = 5000
    HUTCHINSON_SAMPLES = 250

    EXP_DIR = "results/cifar10/resnet20/adam/seed=0/2026-09-02-17-15-51"
    main(EXP_DIR, DATA_SAMPLES, HUTCHINSON_SAMPLES, overwrite=False)

    EXP_DIR = "results/cifar10/resnet20/adam/seed=0/2026-09-08-16-35-55"
    main(EXP_DIR, DATA_SAMPLES, HUTCHINSON_SAMPLES, overwrite=False)

    EXP_DIR = "results/cifar10/resnet20/adamw/seed=0/2026-08-27-21-41-19"
    main(EXP_DIR, DATA_SAMPLES, HUTCHINSON_SAMPLES, overwrite=False)

    EXP_DIR = "results/cifar10/resnet20/adamw/seed=0/2026-09-08-16-35-50"
    main(EXP_DIR, DATA_SAMPLES, HUTCHINSON_SAMPLES, overwrite=False)

    EXP_DIR = "results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-10-34-12"
    main(EXP_DIR, DATA_SAMPLES, HUTCHINSON_SAMPLES, overwrite=False)

    EXP_DIR = "results/cifar10/resnet20/ivadam-decoupled-atmean/seed=0/2026-09-02-16-19-40"
    main(EXP_DIR, DATA_SAMPLES, HUTCHINSON_SAMPLES, overwrite=False)

    # EXP_DIR = "results/cifar10/resnet20/ivadam-decoupled/seed=0/2026-08-27-21-41-17"
    # main(EXP_DIR, DATA_SAMPLES, HUTCHINSON_SAMPLES, overwrite=False)