import math

import torch
from torch import Tensor
from torch.optim.optimizer import (
    _get_scalar_dtype,
    _get_value,
    _to_scalar,
    Optimizer,
    ParamsT,
)
torch.optim.Adam


def vector_to_parameters(vec, parameters) -> None:
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param.data = vec[pointer:pointer+num_param].view_as(param).data
        pointer += num_param


class AdamW(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        weight_decay: float = 0,
        eps: float = 1e-8,
    ) -> None:
        
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not (
            (isinstance(betas[0], float) and isinstance(betas[1], float))
            or (isinstance(betas[0], Tensor) and isinstance(betas[1], Tensor))
        ):
            raise ValueError("betas must be either both floats or both Tensors")
        if isinstance(betas[0], Tensor) and betas[0].numel() != 1:
            raise ValueError("Tensor betas[0] must be 1-element")
        if isinstance(betas[1], Tensor) and betas[1].numel() != 1:
            raise ValueError("Tensor betas[1] must be 1-element")
        betas = (_to_scalar(betas[0]), _to_scalar(betas[1]))

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)
        self._eager_state_init()

    def _eager_state_init(self):
        for group in self.param_groups:
            group["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype(), device="cpu")
            numel = sum((p.numel() for p in group["params"] if p.requires_grad))
            devices = set((p.device for p in group["params"]))
            assert len(devices) == 1
            device = devices.pop()
            group["exp_avg"] = torch.zeros(numel, device=device)
            group["exp_avg_sq"] = torch.zeros(numel, device=device)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            
            state_step = group["step"]
            exp_avg = group["exp_avg"]
            exp_avg_sq = group["exp_avg_sq"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            # State management
            state_step += 1
            state_step = _get_value(state_step)
            # Computation
            flat_grads = torch.cat([p.grad.view(-1) for p in group["params"] if p.requires_grad])
            exp_avg.lerp_(flat_grads, weight=1-beta1)
            exp_avg_sq.mul_(beta2).addcmul_(flat_grads, flat_grads, value=1-beta2)
            del flat_grads
            # Assuming all tensors in the group have been updated for the same number of steps
            bias_correction1 = 1 - beta1**state_step
            bias_correction2_sqrt = math.sqrt(1 - beta2**state_step)
            flat_params = torch.cat([p.data.view(-1) for p in group["params"] if p.requires_grad])
            flat_params.mul_(1-lr*weight_decay).addcdiv_(
                exp_avg,
                exp_avg_sq.sqrt()/bias_correction2_sqrt + eps,
                value=-lr/bias_correction1
            )
            vector_to_parameters(flat_params, group["params"])
            del flat_params

        return loss