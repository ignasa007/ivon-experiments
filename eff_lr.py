from os.path import basename, splitext
import inspect

import torch
import matplotlib.pyplot as plt


optimizer = torch.optim.AdamW((torch.nn.Parameter(torch.randn(1)),), lr=0.002)

# 1. Warmup: 0.0004 -> 0.002 over 5 epochs
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.0004 / 0.002,  # 0.2
    end_factor=1.0,
    total_iters=5,
)
# 2. Cosine annealing: 0.002 -> 0 over 200 epochs
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=200,
    eta_min=0.0,
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[5],
)

lrs = list()
for epoch in range(200):
    lrs.extend([optimizer.param_groups[0]['lr']]*1000)
    scheduler.step()

b2_exp = (1-1e-5) ** torch.arange(1, len(lrs)+1)
plt.plot(lrs, label=r"Original $\eta$")
plt.plot(torch.Tensor(lrs) / b2_exp, label=r"Effective $\eta$ (biased)")
# plt.plot(torch.Tensor(lrs) / b2_exp * (1-b2_exp), label=r"Effective $\eta$ (unbiased)")
plt.tick_params(axis="both", which="major", labelsize=12)
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.xlabel("Step", fontsize=16)
plt.grid()
plt.legend(fontsize=12, framealpha=1.0)

plt.tight_layout()
plt.savefig(f"{splitext(basename(inspect.stack()[0][1]))[0]}.png")