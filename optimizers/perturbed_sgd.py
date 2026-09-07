import torch
from torch import Tensor
from torch.optim.optimizer import (
    _get_scalar_dtype,
    _get_value,
    _to_scalar,
    Optimizer,
    ParamsT,
)


class PerturbedSGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        ess: float,
        lr: float | Tensor = 1e-3,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        weight_decay: float = 0,
        hess_init: float = 0.5,
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
        if not 0.0 < hess_init:
            raise ValueError(f"Invalid hess_init value: {hess_init}")
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
            "ess": ess,
            "hess_init": hess_init,
        }
        super().__init__(params, defaults)
        self._eager_state_init()

    def _eager_state_init(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = torch.tensor(
                            0.0, dtype=_get_scalar_dtype(), device="cpu"
                        )
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

    # Populates the states used by the optimizer to compute the update 
    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(
                        0.0, dtype=_get_scalar_dtype(), device="cpu"
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                exp_avgs.append(state["exp_avg"])
                state_steps.append(state["step"])

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            
            params = []
            grads = []
            exp_avgs = []
            state_steps = []
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            self._init_group(
                group,
                params,
                grads,
                exp_avgs,
                state_steps,
            )
            
            grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
                [params, grads, exp_avgs, state_steps]
            )

            for (params, grads, exp_avgs, state_steps), _ in grouped_tensors.values():
                # State management
                torch._foreach_lerp_(exp_avgs, grads, weight=1-beta1)
                torch._foreach_add_(state_steps, 1)
                rescaled_lrs = [-lr / beta2**_get_value(step) for step in state_steps]
                # Multiplying by 1+rescaled_lr*weight_decay because rescaled_lr = -lr/beta_2^t
                torch._foreach_mul_(params, [1+rescaled_lr*weight_decay for rescaled_lr in rescaled_lrs])
                bias_correction1 = [1-torch.as_tensor(beta1, device=params[0].device)**_get_value(step) for step in state_steps]
                torch._foreach_addcdiv_(params, exp_avgs, bias_correction1, rescaled_lrs)

        return loss

    def store_param_data(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["param_data"] = p.data

    def sample_param_data(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    p.data = state["param_data"] + torch.randn_like(p.data) / (
                        group["ess"] * group["hess_init"] * group["betas"][1]**(state["step"]-1)
                    ).sqrt()

    def restore_param_data(self, clear_data: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    p.data = state["param_data"]
                    if clear_data:
                        del state["param_data"]