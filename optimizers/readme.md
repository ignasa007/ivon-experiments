# Optimizers

The [official implementation of IVON](https://github.com/team-approx-bayes/ivon) had a [critical bug](https://github.com/team-approx-bayes/ivon/issues) in the implementation with `hess_approx=gradsq`; we include the fixed implementation here. We also provide our own implementations of some algorithms:

- **Square-root-free RMSProp** &ndash; proposed in [Lin et al. (2024)](https://proceedings.mlr.press/v235/lin24e.html), but without an open-source implementation, so we write our own.

- **Variational Adam** &ndash; proposed in [Khan et al. (2018)](https://proceedings.mlr.press/v80/khan18a.html), but without an open-source implementation, so we write our own. Take note of the following hyper-parameters:
    1. `decoupled_weight_decay` &ndash; this isn't decoupling the weight decay term from the update, as in AdamW, but from the first-moment accumulator, i.e. `decoupled_weight_decay=False` computes the numerator as $\text{EMA}[\nabla \mathcal{L(\theta)}] + \delta \theta$, but if it is set to `True`, the weight decay contribution is included in the EMA.

- **Perturbed SGD** &ndash; derived from IVON with large initial curvature-estimate $s_0$ and $\beta_2 \approx 1$:
    1. perturb parameters according to an exponentially decreasing variance schedule,
    2. compute gradients,
    3. update original parameters.
    
    Take note of the following hyper-parameters:
    1. `ess` &ndash; the effective sample size is usually set to the number of data samples, but can be set to a smaller value, if appropriate.
    2. `hess_init` &ndash; the initial curvature-estimate from IVON, used for computing the curvature estimate at time $t$ as $\beta_2^{t}s_0$.
    3. `clip_radius` &ndash; element-wise clipping, performed on the preconditioned update.

- **Approximate Perturbed SGD** &ndash; computed as an NGD step with a first-order Taylor approximation of the gradient at the perturbed point:
    $$
        \nabla \mathcal{L}(\theta+\epsilon) \approx \nabla \mathcal{L}(\theta) + \nabla^2 \mathcal{L}(\theta) \epsilon
    $$
    That is, we simply peturb the parameters after performing a standard update.

    **TODO:** Add element-wise clipping, and write a faster implementation.