from common.trainutils import loadcheckpoint
import os

d = "results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-10-34-12"
for fn in sorted(os.listdir(d)):
    if fn.startswith("checkpoint") and fn.endswith(".pt"):
        _, model, opt, scheduler, _ = loadcheckpoint(f"results/cifar10/resnet20/ivon-price/seed=0/2026-08-18-10-34-12/{fn}")
        for group in opt.param_groups:
            print(fn)
            print(group["hess"].sort().values)
            print((group["ess"]*(group["hess"]+group["weight_decay"])).sort(descending=True).values ** (-0.5))
            print()