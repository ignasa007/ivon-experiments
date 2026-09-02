import os
import torch
from common.trainutils import loadcheckpoint

d = "results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-10-34-12"
for fn in sorted(os.listdir(d)):
    if fn.startswith("checkpoint") and fn.endswith(".pt") and fn != "checkpoint.pt":
        _, model, opt, _, _ = loadcheckpoint(f"{d}/{fn}")
        for group in opt.param_groups:
            print(fn)
            print(group["hess"].sort().values)
            print((group["ess"]*(group["hess"]+group["weight_decay"])).sort(descending=True).values ** (-0.5))
        print()

print("#"*40 + "\n")

d = "results/cifar10/resnet20/ivadam-decoupled/seed=0/2026-08-27-21-41-17"
for fn in sorted(os.listdir(d)):
    if fn.startswith("checkpoint") and fn.endswith(".pt") and fn != "checkpoint.pt":
        _, model, opt, _, _ = loadcheckpoint(f"{d}/{fn}")
        for group in opt.param_groups:
            print(fn)
            hess = torch.cat([
                opt.state[p]["exp_avg_sq"].flatten()
                for p in group["params"] if p.requires_grad
            ]).sqrt().sort().values
            print(hess)
            print((group["ess"]*(hess+group["weight_decay"])) ** (-0.5))
        print()