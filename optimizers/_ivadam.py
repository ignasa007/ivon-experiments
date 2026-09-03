from contextlib import contextmanager

import torch
from torch import Tensor
from torch.optim.optimizer import (
    _default_to_fused_or_foreach,
    _get_scalar_dtype,
    _get_value,
    _to_scalar,
    Optimizer,
    ParamsT,
)


class IVAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        ess: float,
        lr: float | Tensor = 1e-3,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        weight_decay: float = 0,
        decoupled_wd: bool = False, 
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
            "decoupled_wd": decoupled_wd,
            "ess": ess,
        }
        super().__init__(params, defaults)
        self._eager_state_init()

    # This is implemented to have non-empty `exp_avg_sq`,
    # so as to allow for sampling before the first step is taken
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
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

    # Set the state of an optimizer to a user-specified value
    # We'll probably never use it
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("decoupled_wd", False)
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
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
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
            
            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                decoupled_wd=group["decoupled_wd"],
            )

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
                    # Can debias `state["exp_avg_sq"]` here but it barely changes
                    # the quality of approximation to Hessian in the toy examples
                    p.data = state["param_data"] + torch.randn_like(p.data) / (
                        group["ess"]*(state["exp_avg_sq"].sqrt()+group["weight_decay"])
                    ).sqrt()

    def restore_param_data(self, clear_data: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    p.data = state["param_data"]
                    if clear_data:
                        del state["param_data"]

def _single_tensor_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float | Tensor,
    beta2: float | Tensor,
    lr: float | Tensor,
    weight_decay: float,
    decoupled_wd: bool,
) -> None:

    for param, grad, exp_avg, exp_avg_sq, state_step in zip(
        params, grads, exp_avgs, exp_avg_sqs, state_steps
    ):

        if decoupled_wd or weight_decay == 0:
            # IVON-like implementation, where grad of log-prior is not included in the first moment
            # https://arxiv.org/pdf/2402.17641, Algorithm 1, Lines 4 and 7
            exp_avg.lerp_(grad, weight=1-beta1)
        else:
            # VAdam-like implementation, where grad of log-prior is included in the first moment
            # https://arxiv.org/pdf/1806.04854, Figure 5, VAdam, Lines 5 and 8
            exp_avg.lerp_(grad.add(param, alpha=weight_decay), weight=1-beta1)

        # We save memory by bypassing materialization; `addcmul` needs float `value`, but it can
        # handle 0D tensors, which we construct using `_to_scalar` at the start of this method
        exp_avg_sq.mul_(beta2)
        exp_avg_sq.addcmul_(grad, grad, value=1-beta2)

        state_step += 1
        bias_correction1 = 1 - beta1**_get_value(state_step)
        bias_correction2 = 1 - beta2**_get_value(state_step)

        denom = (exp_avg_sq/bias_correction2).sqrt().add(weight_decay)
        
        step_size = -lr/bias_correction1
        if not decoupled_wd or weight_decay == 0:
            param.addcdiv_(exp_avg, denom, value=step_size)
        else:
            num = exp_avg.add(param, alpha=weight_decay*bias_correction1)
            param.addcdiv_(num, denom, value=step_size)

def _multi_tensor_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float | Tensor,
    beta2: float | Tensor,
    lr: float | Tensor,
    weight_decay: float,
    decoupled_wd: bool,
) -> None:
    
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps]
    )

    for (params, grads, exp_avgs, exp_avg_sqs, state_steps), _ in grouped_tensors.values():

        if decoupled_wd or weight_decay == 0:
            torch._foreach_lerp_(exp_avgs, grads, weight=1-beta1)
        else:
            torch._foreach_lerp_(
                exp_avgs,
                torch._foreach_add(grads, params, alpha=weight_decay),
                weight=1-beta1
            )

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1-beta2)

        torch._foreach_add_(state_steps, 1)
        bias_correction1 = [1 - torch.as_tensor(beta1, device=params[0].device)**_get_value(step) for step in state_steps]
        bias_correction2 = [1 - beta2**_get_value(step) for step in state_steps]
        bias_correction2_sqrt = [bc**0.5 for bc in bias_correction2]

        denoms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denoms, bias_correction2_sqrt)
        torch._foreach_add_(denoms, weight_decay)

        step_sizes = torch.Tensor([-lr/bc for bc in bias_correction1])
        if not decoupled_wd or weight_decay == 0:
            torch._foreach_addcdiv_(params, exp_avgs, denoms, step_sizes)
        else:
            nums = torch._foreach_addcmul(exp_avgs, params, bias_correction1, weight_decay)
            torch._foreach_addcdiv_(params, nums, denoms, step_sizes)

def adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float | Tensor,
    beta2: float | Tensor,
    lr: float | Tensor,
    weight_decay: float,
    decoupled_wd: bool,
) -> None:

    _, foreach = _default_to_fused_or_foreach(
        params, differentiable=False, use_fused=False
    )
    if foreach is None:
        func = _single_tensor_adam
    else:
        func = _multi_tensor_adam

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        decoupled_wd=decoupled_wd,
    )