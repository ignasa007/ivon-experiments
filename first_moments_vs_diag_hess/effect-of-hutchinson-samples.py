import os

import numpy as np
from scipy.optimize import curve_fit

from main import *
from utilities import *


HUTCHINSON_SAMPLES = [500, 1000, 2000, 5000]
HUTCHINSON_LABELS = ["$S = 500$", "$S = 1$k", "$S = 2$k", "$S = 5$k"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]


def estimate_hess_diag(X, Y, model, loss_type, optimizer, device):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sum_samples = 0.
    hess_diags = list()
    for i in range(1, max(HUTCHINSON_SAMPLES)+1):
        random_vector = torch.randint(2, (num_params,), device=device, dtype=X.dtype).mul(2).sub(1)
        sum_samples += random_vector * compute_hvp(X, Y, model, random_vector, loss_type, device)
        if i in HUTCHINSON_SAMPLES:
            hess_diags.append(sum_samples/i)
    return torch.stack(hess_diags)


def run(args, ckpts):

    Dataset, output_dim = dataset_map[args.dataset]
    args.batch_size = args.batch_size if args.batch_size is not None else args.subset_size
    Model = model_map[args.model]
    widths = as_list(args.widths)
    model_kwargs = dict(bias=args.bias, normalization=args.normalization)
    loss_type = loss_type_map[args.loss_type]
    Optimizer = optimizer_map[args.optimizer]
    optim_kwargs = make_optim_kwargs(args)
    tracker_fns = (compute_exp_avg_sq, estimate_hess_diag)
    device = torch.device(args.device)

    assets, out = train_gd(
        Dataset=Dataset, output_dim=output_dim,
        subset_size=args.subset_size, batch_size=args.batch_size,
        Model=Model, widths=widths, model_kwargs=model_kwargs,
        act_name=args.activation, act_kwargs=dict(),    # don't currently support `act_kwargs`
        loss_type=loss_type, Optimizer=Optimizer, optim_kwargs=optim_kwargs,
        steps=args.steps, log_every=None, ckpts=ckpts,
        device=device, seed=args.seed, tracker_fns=tracker_fns,
    )

    return assets, out


def main(cmd: str, ckpts: list[int], save_fn: str) -> None:

    args = parser.parse_args(cmd.split())
    _, (_, xs, ys) = run(args, ckpts)
    fig, axs = plt.subplots(2, 3, figsize=(7.5*3, 4.5*2))
    
    for i, (ckpt, ax) in enumerate(zip(ckpts, axs.flatten())):
    
        for j, (s, color) in enumerate(zip(HUTCHINSON_LABELS, COLORS)):
    
            x, y = np.asarray(xs[i]), np.asarray(ys[i,j])
            mask = (x > 1e-24) & (y > 1e-24)
            x, y = x[mask], y[mask]
            ax.scatter(x, y, s=1, color=color, alpha=0.1)
    
            if sum(mask) >= 2:
                p0 = [0., 75., 0.5]
                bounds = (0, np.inf)
                opt, _ = curve_fit(power_law_offset, x, np.log(y), p0=p0, bounds=bounds, maxfev=10000)
                x_fit = np.geomspace(x.min(), x.max(), 200)
                y_fit = np.exp(power_law_offset(x_fit, *opt))
                ax.plot(
                    x_fit, y_fit, linewidth=5, linestyle="--", color=color,
                    label=(rf"{s}, " if i==0 else "") + rf"$\gamma = {opt[-1]:.2f}$"
                )
    
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
        ax.legend(fontsize=18, framealpha=1, ncol=2, loc="lower right")
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.set_title(f"Checkpoint {ckpt}", fontsize=20)
    
    for ax in axs[-1,:]:
        ax.set_xlabel("Exp Avg Sq", fontsize=20)
    for ax in axs[:,0]:
        ax.set_ylabel("Hessian Diagonal", fontsize=20)

    fig.tight_layout()
    if save_fn is not None:
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        plt.savefig(save_fn)
    plt.close()


if __name__ == "__main__":

    STEPS = 1000
    CKPTS = (np.array([1, 3, 5, 7, 9, 10])*(STEPS//10)).tolist()
    save_fn = f"{ASSETS}/implementation-checks/effect-of-hutchinson-samples.png"
    cmd = f"--optimizer adam --widths [64]*3 --hutchinson_samples {max(HUTCHINSON_SAMPLES)} --steps {STEPS}"
    print(f"*** python main.py {cmd} ***")
    main(cmd, CKPTS, save_fn)