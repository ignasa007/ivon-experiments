import torch
from torch import Tensor
from torch.optim.optimizer import (
    _get_scalar_dtype,
    _to_scalar,
    Optimizer,
    ParamsT,
)


class SFRMSProp(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        weight_decay: float = 0,
        batch_size: int = 1,
        eps: float = 5e-4,
    ) -> None:
        
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter atw index 0: {betas[0]}")
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
            "batch_size": batch_size,
            "eps": eps,
        }
        super().__init__(params, defaults)

    # Set the state of an optimizer to a user-specified value
    # We'll probably never use it
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=_get_scalar_dtype())

    # Populates the states used by the optimizer to compute the update 
    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
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
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.ones_like(p)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )
            
            grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
                [params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps]
            )

            for (params, grads, exp_avgs, exp_avg_sqs, state_steps), _ in grouped_tensors.values():

                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=(1-beta2)*group["batch_size"])

                torch._foreach_mul_(exp_avgs, beta1)
                torch._foreach_addcdiv_(exp_avgs, grads, torch._foreach_add(exp_avg_sqs, group["eps"]))
                if group["weight_decay"] != 0.:
                    torch._foreach_add_(exp_avgs, params, alpha=group["weight_decay"])

                torch._foreach_sub_(params, exp_avgs, alpha=group["lr"])

        return loss