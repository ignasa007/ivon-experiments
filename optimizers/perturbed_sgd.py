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


def vector_to_parameters(vec, parameters) -> None:
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param.data = vec[pointer:pointer+num_param].view_as(param).data
        pointer += num_param


class PerturbedSGDOld(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        weight_decay: float = 0,
        ess: float = None,
        hess_init: float = 0.5,
        clip_radius: float = torch.inf,
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
        if not isinstance(ess, float) or not ess > 0.:
            raise ValueError(f"Invalid ess value: {ess}")
        if not 0.0 < hess_init:
            raise ValueError(f"Invalid hess_init value: {hess_init}")
        if not 0.0 < clip_radius:
            raise ValueError(f"Invalid clip_radius value: {clip_radius}")
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
            "clip_radius": clip_radius,
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
                    state["exp_avg"] = torch.zeros_like(p)
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
            hess_init = group["hess_init"]
            clip_radius = group["clip_radius"]

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
                torch._foreach_add_(state_steps, 1)
                # Computation
                torch._foreach_lerp_(exp_avgs, grads, weight=1-beta1)
                bias_correction1 = [1-torch.as_tensor(beta1, device=params[0].device)**_get_value(step) for step in state_steps]
                updates = torch._foreach_addcmul(exp_avgs, params, bias_correction1, value=weight_decay)
                # Clamp
                scalings = [bc1 * beta2**step for bc1, step in zip(bias_correction1, state_steps)]
                if math.isfinite(clip_radius):
                    clip_radii = [clip_radius*hess_init*_get_value(scaling) for scaling in scalings]
                    torch._foreach_clamp_max_(updates, clip_radii)
                    torch._foreach_clamp_min_(updates, [-1.*clip_radius for clip_radius in clip_radii])
                # Update
                torch._foreach_addcdiv_(params, updates, scalings, value=-lr)

        return loss

    @torch.no_grad()
    def store_param_data(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["data"] = p.data.clone()

    # NOTE: slightly different numbers, maybe because of GPU
    @torch.no_grad()
    def sample_param_data(self):
        for group in self.param_groups:
            params = [p for p in group["params"] if p.requires_grad]
            if not params:
                continue
            params_data = [self.state[p]["data"] for p in params]
            # We can do-away with this if we're using only one MC sample
            # TODO: This works only when we use one sample since it links params and params_data -- MAYBE, IDK
            torch._foreach_copy_(params, params_data)
            noises = [torch.randn_like(p) for p in params]
            init_scale, beta2 = group["ess"]*group["hess_init"], group["betas"][1]
            scales = [(init_scale*beta2**self.state[p]["step"])**(-0.5) for p in params]
            torch._foreach_addcmul_(params, noises, scales)

    @torch.no_grad()
    def restore_param_data(self, clear_data: bool = False):
        for group in self.param_groups:
            params = [p for p in group["params"] if p.requires_grad]
            if not params:
                continue
            params_data = [self.state[p]["data"] for p in params]
            torch._foreach_copy_(params, params_data)
            if clear_data:
                for p in params:
                    del self.state[p]["data"]


class PerturbedSGDNew(PerturbedSGDOld):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        weight_decay: float = 0,
        ess: float = None,
        hess_init: float = 0.5,
        clip_radius: float = torch.inf,
    ) -> None:

        super().__init__(params, lr, betas, weight_decay, ess, hess_init, clip_radius)

    def _eager_state_init(self):
        for group in self.param_groups:
            group["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype(), device="cpu")
            numel = sum((p.numel() for p in group["params"] if p.requires_grad))
            devices = set((p.device for p in group["params"]))
            assert len(devices) == 1
            group["exp_avg"] = torch.zeros(numel, device=devices.pop())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            state_step = group["step"]
            exp_avg = group["exp_avg"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            hess_init = group["hess_init"]
            clip_radius = group["clip_radius"]

            # State management
            state_step += 1
            state_step = _get_value(state_step)
            # Computation
            flattened_grads = torch.cat([p.grad.view(-1) for p in group["params"] if p.requires_grad])
            exp_avg.lerp_(flattened_grads, weight=1-beta1)
            del flattened_grads
            # Assuming all tensors in the group have been updated for the same number of steps
            bias_correction1 = 1 - beta1 ** state_step
            flattened_params = torch.cat([p.data.view(-1) for p in group["params"] if p.requires_grad])
            updates = torch.add(exp_avg, flattened_params, alpha=bias_correction1*weight_decay)
            # Clamp
            scaling = bias_correction1 * beta2 ** state_step
            if math.isfinite(clip_radius):
                clip_radii = clip_radius * hess_init * scaling
                updates.clamp_(-clip_radii, clip_radii)
            flattened_params.sub_(updates, alpha=lr/scaling)
            del updates
            # Update
            vector_to_parameters(flattened_params, group["params"])
            del flattened_params

        return loss

    @torch.no_grad()
    def store_param_data(self):
        for group in self.param_groups:
            group["data"] = torch.cat([p.data.view(-1) for p in group["params"]])

    # NOTE: slightly different numbers, but because we sample at once instead of separately for each parameter
    @torch.no_grad()
    def sample_param_data(self):
        for group in self.param_groups:
            flattened_params = group["data"]
            noise = torch.randn_like(flattened_params)
            precision = group["ess"] * group["hess_init"] * group["betas"][1]**group["step"]
            flattened_params = flattened_params.add(noise, alpha=precision**(-0.5))
            vector_to_parameters(flattened_params, group["params"])

    @torch.no_grad()
    def restore_param_data(self, clear_data: bool = False):
        for group in self.param_groups:
            vector_to_parameters(group["data"], group["params"])
            if clear_data:
                del group["data"]


PerturbedSGD = PerturbedSGDNew