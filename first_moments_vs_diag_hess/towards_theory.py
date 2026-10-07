import os
from functools import partial

from torch.func import functional_call, vjp
from torchvision import datasets
from torch import optim

from constants import *
from utilities import MLP, compute_gradient, compute_exp_avg_sq, train_gd, plot
from check_ggn_vs_hess import compute_ggn_diag, compute_hess_diag


OUTPUT_DIM = 10
MODEL, WIDTHS, MODEL_KWARGS = MLP, [128]*0, dict(bias=False)
ACTIVATION, ACT_KWARGS = "Linear", dict()
LOSS_TYPE = "Mean Squared Error"
OPTIMIZER, OPTIM_KWARGS = optim.Adam, dict(lr=1e-3, betas=(0.9, 0.95))
EPOCHS = 1000; LOG_EVERY = EPOCHS // CKPTS


def compute_residual_vjp(X, Y, model, loss_type, optimizer, device):
    def f(params):
        return functional_call(model, params, (X,))
    names = [name for name, _ in model.named_parameters()]
    params = dict(model.named_parameters())
    _, vjp_fn = vjp(f, params)
    Jtv = vjp_fn(model(X)-Y)[0]
    Jtv = 1/X.size(0) * torch.cat([Jtv[name].reshape(-1) for name in names])
    return Jtv.abs()


assets, out = train_gd(
    Dataset=datasets.MNIST, output_dim=OUTPUT_DIM, subset_size=SUBSET_SIZE,
    Model=MODEL, widths=WIDTHS, model_kwargs=MODEL_KWARGS,
    act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
    loss_type=LOSS_TYPE, Optimizer=OPTIMIZER, optim_kwargs=OPTIM_KWARGS,
    epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED,
    tracker_fns=(
        compute_gradient,
        compute_exp_avg_sq,
        partial(compute_ggn_diag, hutchinson_samples=HUTCHINSON_SAMPLES),
        partial(compute_hess_diag, hutchinson_samples=HUTCHINSON_SAMPLES),
        compute_residual_vjp,
    )
)

losses, gradients, exp_avg_sqs, ggn_diags, hess_diags, residual_vjp = out
ckpts = [1, 3, 5, 7, 9, 10]
SAVE_DIR = "towards-theory"
os.makedirs(SAVE_DIR, exist_ok=True)

xlabel = "Instantaneous Grad Sq"
ylabel = "GGN Diagonal"
gradients_sq = [g**2 for g in gradients]
plot(gradients_sq, ggn_diags, ckpts, LOG_EVERY, xlabel, ylabel, save_fn=f"{SAVE_DIR}/instsq-vs-ggn.png")

xlabel = "Exp Avg Sq"
ylabel = "GGN Diagonal"
plot(exp_avg_sqs, ggn_diags, ckpts, LOG_EVERY, xlabel, ylabel, save_fn=f"{SAVE_DIR}/expavgsq-vs-ggn.png")

xlabel = "Exp Avg Sq"
ylabel = "Hessian Diagonal"
plot(exp_avg_sqs, hess_diags, ckpts, LOG_EVERY, xlabel, ylabel, save_fn=f"{SAVE_DIR}/expavgsq-vs-hess.png")

xlabel = "Exp Avg Sq"
ylabel = "Residual VJP"
plot(exp_avg_sqs, residual_vjp, ckpts, LOG_EVERY, xlabel, ylabel, save_fn=f"{SAVE_DIR}/expavgsq-vs-residual-vjp.png")

xlabel = "Residual VJP"
ylabel = "GGN Diagonal"
plot(residual_vjp, ggn_diags, ckpts, LOG_EVERY, xlabel, ylabel, save_fn=f"{SAVE_DIR}/residual-vjp-vs-ggn.png")