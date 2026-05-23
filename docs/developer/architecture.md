# Developer Architecture

This document records implementation conventions for contributors. User-facing
workflow documentation lives in the top-level guide pages.

The code is split into numerical kernels, variational nodes, and a top-level
model builder so graph structure lives above the node implementations.

## Numerical Conventions

- CTMC generators are column-sum-zero matrices with entries
  `Q[to_state, from_state]`.
- Transition rates are ordered by row-major off-diagonal entries. For three
  states this is `[[-, 0, 1], [2, -, 3], [4, 5, -]]`.
- Arrays used by kernels and node moments are `float32`; JAX x64 should remain
  disabled.
- Missing MS2 observations are represented by `NaN` and a boolean finite mask.
  Kernels sanitize missing values before residuals are formed, so missing data
  does not create `NaN` gradients or posterior moments.
- Pol2 loading intervals also carry an internal support mask derived from the
  finite observations and the MS2 kernel representation. Unsupported intervals
  may report prior marginals for inspection, but they do not send reverse
  potentials to `PromoterState` and do not contribute loading-rate sufficient
  statistics.

## Core Kernels

`viprodyne.core.rate_edges` owns transition indexing and generator wrapping.
Use `RateEdge` metadata to connect a transition-rate node to a promoter edge.
If `RateEdge.drive_node` is set, the promoter node treats the edge as driven by
a time-varying contact probability.

`viprodyne.core.tilted_ctmc` solves piecewise-constant tilted CTMCs using JAX.
The tilted operator on an interval is

```text
Q_tilde + diag(potential)
```

where `Q_tilde` carries jump log-rates through its off-diagonal entries and the
diagonal potential carries local Feynman-Kac corrections.

`viprodyne.core.bernoulli_transfer_pol2` contains the fixed-grid Bernoulli Pol2
loading kernels:

- `mean_field_bernoulli_elbo_terms` and `mean_field_bernoulli_elbo` expose
  separate data, variance, normalization, prior, and entropy terms.
- `exact_bernoulli_posterior` enumerates all binary loading configurations for
  small systems and tests.
- `bernoulli_transfer_log_likelihood` computes exact loading log evidence with a
  sliding-window transfer algorithm and does not materialize a dense
  observation-by-loading matrix.
- `bernoulli_transfer_log_likelihood_batch` vmaps that exact transfer pass over
  independent trajectories for large same-grid trace sets.
- `bernoulli_transfer_posterior` computes exact transfer marginals and entropy
  from `log_Z` derivative identities. It avoids a materialized joint posterior,
  but it is more expensive than the log-likelihood-only transfer pass.

`viprodyne.core.ms2_kernels` owns the public MS2 kernel specification and the
internal conversion from user timing inputs to Pol2 observation representations:

- `ProximalKernel` is the current built-in parameterized kernel. It uses
  `t_rise`, `t_plateau`, and `rna_intensity`.
- `ModelConfig.ms2_kernel` can also take `"proximal"` or a custom
  JAX-compatible kernel function.
- `build_ms2_observation_model` derives the dense, transfer, or sampler
  representation used by `PolymeraseLoadings`. The transfer path extracts
  row-specific windows from the kernel without storing a full dense
  observation-by-loading matrix.

`viprodyne.core.pol2_sampler` contains the continuous-time reversible-jump Pol2
loading sampler and thermodynamic-integration log-partition estimator. It is
available through `PolymeraseLoadings(mode="sampler")` and
`ModelConfig.pol2_mode="sampler"` for proximal MS2 kernels.

The dense design matrix and transfer-window representation are internal objects.
The user-facing model API takes observed intensities, timing, and an MS2 kernel
specification. Dense mode is still useful internally for tiny exact enumeration
tests and generic mean-field checks, but it scales as
`O(n_observations * n_loadings)`. For regular MS2 kernels, the model builder
derives the transfer representation so memory stays tied to kernel support and
grid size instead of a full dense convolution matrix.

`viprodyne.core.contact_survival` implements the corrected contact-survival
profile for driven rates. The discrete survival factor is

```text
1 - p_contact(t) * (1 - exp(-k * dt))
```

and the MAP profile uses the corresponding survival log term rather than the
older linear exposure approximation.

## Variational Nodes

All nodes inherit from `VariationalNode` and implement:

- `moments()`
- `update(context)`
- `entropy()`
- `elbo_contribution()`
- `sample()`

Graph connectivity is held by `VariationalGraph`, not by the nodes. The graph
passes parent and child names through `UpdateContext`, so shared parameters and
dataset plates can be changed at graph-construction time.

Parameter nodes:

- `InitialStateProb`: Dirichlet node for per-dataset initial promoter state
  probabilities.
- `LoadingRate`: Gamma node for state-specific Pol2 loading rates.
- `TransitionRate`: Gamma node for ordinary promoter transition rates.
- `DrivenRateMap`: bounded MAP node for contact-driven rates using the
  contact-survival profile.
- `RcNode`: MAP or pinned node that emits `p_contact` for driven transitions.

Domain nodes:

- `ObservedIntensity`: deterministic observed MS2 intensity node.
- `PromoterState`: tilted CTMC node for `q(s_t)`.
- `PolymeraseLoadings`: Pol2 loading node with `mean_field`, `exact`, and
  `transfer` modes. The model builder supplies the internal dense or transfer
  representation from the configured MS2 kernel.

Pinned parameter nodes emit deterministic moments and have zero entropy.

`PolymeraseLoadings` does not take a dataset-level Pol2 prior in model-built
graphs. Its Bernoulli loading prior is derived during updates from promoter
interval state probabilities and loading-rate moments using the same natural
CAVI terms as the reverse promoter message:

```text
log p_i^* = sum_s q_i(s) * E[log(1 - exp(-r_s * dt_i))]
log (1 - p_i^*) = sum_s q_i(s) * (-E[r_s] * dt_i)
P(tau_i = 1) = softmax(log p_i^*, log (1 - p_i^*))
```

For Gamma loading-rate nodes, `E[log(1 - exp(-r_s * dt_i))]` is evaluated with
the convergent log-survival series. This is intentionally not the arithmetic
expectation `E[1 - exp(-r_s * dt_i)]`, because the coordinate update depends on
expected log factors.

The promoter update consumes the reverse message from `PolymeraseLoadings` as an
interval state potential:

```text
q(tau_i) * E[log(1 - exp(-r_s * dt_i))]
  + (1 - q(tau_i)) * (-E[r_s] * dt_i)
```

divided by `dt_i` before being added to the tilted CTMC potential. Loading-rate
sufficient statistics use the mean-field product `q(tau_i) * q_i(s)`.

The promoter initial condition uses `E[log pi]`, normalized into a valid initial
probability vector for the CTMC solver:

```text
q(s_0) proportional to exp(E[log pi])
```

The initial Pol2 prior inside the node is only a placeholder until graph
messages from `PromoterState` and `LoadingRate` are available.

For sampler mode, `PolymeraseLoadings` emits a posterior loading rate and
expected loading counts on the sampler fine grid. Its promoter-state reverse
message uses the Poisson point-process form,

```text
E[n_i] * E[log r_s] - E[r_s] * dt_i
```

divided by `dt_i` before being added to the tilted CTMC potential. The sampler
prior intensity is `exp(sum_s q_i(s) * E[log r_s])`, matching the natural CAVI
message rather than the arithmetic rate average.

Missing data is handled at the loading-grid level. `PolymeraseLoadings` derives
`loading_mask` internally from `finite_mask` plus either the dense design
matrix, transfer windows, or sampler kernel support. The mask gates
Pol2-to-promoter potentials and loading-rate counts/exposures, preventing
prior-only loadings from updating rates when nearby MS2 observations are absent.

Current limitation: batched traces within a dataset plate must share the same
time grid, sampling times, and MS2 kernel representation. Heterogeneous trace
timing should be split across separate `MS2Dataset` plates.

## Promoter-State Construction

For each transition edge, `PromoterState` builds `Q_tilde` and the diagonal
potential from parent moments.

For an ordinary rate,

```text
Q_tilde[to, from] = exp(E[log k])
potential[from] += exp(E[log k]) - E[k]
```

This gives the correct jump term and the correct expected local exit term in
the mean-field path update.

For a driven rate with contact probability `p(t)`,

```text
Q_tilde[to, from, t] = p(t) * exp(E[log k])
effective_exit(t) = -log(1 - p(t) * (1 - exp(-E[k] * dt))) / dt
potential[from, t] += Q_tilde[to, from, t] - effective_exit(t)
```

The node also emits `contact_survival_stats_by_rate`, keyed by rate-node name,
so each `DrivenRateMap` consumes only the sufficient statistics for its own
transition.

## Top-Level Model Builder

Create one {class}`viprodyne.MS2Dataset` per dataset plate and pass them to
{class}`viprodyne.ViprodyneModel` with a {class}`viprodyne.ModelConfig`.

`MS2Dataset` contains observed intensities with shape
`(n_traces, n_timepoints)`, noise, optional `dt` or `sampling_times`, and an
optional missing-data mask. Contact drives are model inputs, not data fields.
Single-trace datasets are passed as `(1, n_timepoints)`. Dataset names are
optional on the data object; `ViprodyneModel` assigns stable names such as
`dataset_0` to unnamed datasets and rejects duplicate explicit names.
It does not require users to pass dense design matrices or transfer windows.
Those are derived internally from the MS2 kernel in `ModelConfig`.

For fitting many traces, put all same-grid traces for an experimental dataset
or condition into one `MS2Dataset`. Rate scopes are controlled by
`ModelConfig.transition_rate_scope` and `ModelConfig.loading_rate_scope`:

- `"track"` creates one vector-valued rate node with one entry per trace in the
  dataset plate;
- `"dataset"` creates one rate node per `MS2Dataset.rate_group` when supplied,
  otherwise one rate node per `MS2Dataset.name`;
- `"global"` creates one shared rate node for the whole model.

Individual rates can override the default scope with
`transition_rate_scopes={rate_index: scope}` and
`loading_rate_scopes={state_index: scope}`. This supports mixed models such as a
global `k_on`, dataset-level `k_off`, and per-track loading rates in the same
graph. The boolean shortcuts `shared_transition_rates`, `shared_loading_rates`,
`shared_transition_rate_indices`, and `shared_loading_rate_states` options are
kept as aliases for global sharing.

`MS2Dataset` does not contain `prior_load_probabilities`. In model-built graphs,
those are derived from `PromoterState` and `LoadingRate` parents.

```python
import numpy as np

from viprodyne import MS2Dataset, ModelConfig, ViprodyneModel

dataset = MS2Dataset(
    observed=np.array([[0.1, np.nan, 0.8]], dtype=np.float32),
    noise_std=np.float32(0.5),
    dt=np.float32(0.5),
)

config = ModelConfig(
    n_states=2,
    shared_transition_rates=False,
    shared_loading_rates=False,
    pol2_mode="auto",
    ms2_kernel="proximal",
    t_rise=np.float32(0.25),
    t_plateau=np.float32(0.75),
    rna_intensity=np.float32(1.0),
)

model = ViprodyneModel(datasets=(dataset,), config=config)
fit = model.run_inference(max_iterations=100, tolerance=1e-4)
condition = fit.datasets["dataset_0"]

state_posterior = condition.state_posterior
loading_posterior = condition.loading_posterior
predicted_ms2 = condition.predicted_signal
transition_rates = condition.transition_rates
loading_rates = condition.loading_rates
```

The standard inference entry point is
{meth}`viprodyne.ViprodyneModel.run_inference` or the alias
{meth}`viprodyne.ViprodyneModel.fit`. It returns
{class}`viprodyne.ModelInferenceResult`, which contains the CAVI diagnostics
and one {class}`viprodyne.DatasetInferenceResult` per dataset plate. The result
fields expose posterior arrays and fitted rates without requiring users to look
up graph node names.

`model.fit_cavi(...)` and `viprodyne.fit.run_cavi(...)` remain available for
lower-level schedule testing. The default CAVI schedule updates hidden
promoter/Pol2 nodes first, then parameter nodes. Convergence is monitored from
the maximum relative change in parameter-node values, and the model ELBO is
computed only once after the final sweep because Pol2 ELBO terms can be
expensive.

Each `DatasetInferenceResult` contains:

- `state_posterior`: promoter state posterior on `state_posterior_times`,
  shaped `(n_traces, n_state_times, n_states)`;
- `loading_posterior`: Pol2 loading posterior on `loading_posterior_times`,
  shaped `(n_traces, n_loading_times)`;
- `loading_posterior_rate`: posterior Pol2 loading rate for sampler mode;
- `predicted_signal`: posterior mean MS2 intensity at observation times;
- `loading_mask`: loading-grid support mask derived from missing observations;
- `initial_probabilities`, `transition_rates`, and `loading_rates`: fitted
  parameter moments, with transition-rate keys following the documented
  off-diagonal ordering and loading-rate keys following promoter state index.

Final ELBO accounting is factor-based. Gamma and Dirichlet parameter nodes
contribute expected log prior plus entropy, `PolymeraseLoadings` contributes its
loading/data log partition, and `PromoterState` contributes the promoter path
factor plus path entropy. The promoter node subtracts its Pol2 child potential
from the tilted CTMC log partition so the Pol2 loading factor is not counted
twice.

```python
fit = model.run_inference(max_iterations=100, tolerance=1e-4, compute_elbo=True)
print(fit.cavi.converged, fit.cavi.max_parameter_change, fit.cavi.elbo)
```

Regularly sampled datasets should pass `dt`; irregularly sampled datasets
should pass `MS2Dataset.sampling_times`. The model derives its internal loading
intervals from those public timing inputs.

`ModelConfig.ms2_kernel` can be a named kernel string, a `ProximalKernel`
instance, or a custom JAX-compatible callable. The current built-in
parameterized kernel is `"proximal"` and uses `t_rise`, `t_plateau`, and
`rna_intensity`.

`ModelConfig.pol2_mode="auto"` chooses the transfer backend from the kernel
configuration. `pol2_mode="mean_field"` or `"exact"` asks the builder to create
the internal dense representation and should be reserved for small tests and
diagnostics. `pol2_mode="sampler"` uses the continuous Pol2 sampler for
proximal kernels; tune `sampler_iterations`, `sampler_repeats`, and
`sampler_fine_grid`. Sampler ELBOs can use thermodynamic integration with
`sampler_compute_elbo=True`, or a final mean-field diagnostic term with
`pol2_elbo_mode="mean_field"`.

`run_inference`, `fit`, `inference_result`, and `dataset_result` accept
`posterior_times`, plus separate `state_times` and `loading_times` overrides.
These affect only returned posterior arrays. MS2 signal predictions remain on
the observation times.

Example with heterogeneous sampling times:

```python
d0 = MS2Dataset(
    name="d0",
    observed=np.array([[0.1, 0.8]], dtype=np.float32),
    noise_std=np.float32(0.5),
    sampling_times=np.array([0.25, 0.75], dtype=np.float32),
)
d1 = MS2Dataset(
    name="d1",
    observed=np.array([[0.2, 0.5, 0.9]], dtype=np.float32),
    noise_std=np.float32(0.5),
    sampling_times=np.array([0.125, 0.5, 1.125], dtype=np.float32),
)
config = ModelConfig(
    n_states=2,
)
model = ViprodyneModel(datasets=(d0, d1), config=config)
```

Example with a custom kernel:

```python
import jax.numpy as jnp

def rectangular_kernel(time_offsets):
    return jnp.where((time_offsets >= 0.0) & (time_offsets < 0.75), 1.0, 0.0)


config = ModelConfig(
    n_states=2,
    ms2_kernel=rectangular_kernel,
)
model = ViprodyneModel(
    datasets=(dataset,),
    config=config,
)
```

Driven transitions are selected by transition index. For a two-state model,
index `1` is `0 -> 1`. Contact-drive inputs are stored in `ModelConfig` as a
dataset-ordered tuple.

```python
dataset = MS2Dataset(
    observed=np.array([[0.1, 0.8]], dtype=np.float32),
    noise_std=np.float32(0.5),
    dt=np.float32(0.5),
)

def fixed_contact(rc):
    del rc
    return np.array([0.25, 0.75], dtype=np.float32)


config = ModelConfig(
    n_states=2,
    driven_transition_indices=(1,),
    contact_drives=(fixed_contact,),
    driven_rate_initial=np.float32(0.8),
    driven_rate_bounds=(1e-4, 10.0),
)

model = ViprodyneModel(datasets=(dataset,), config=config)
```

Driven transition rates can be scoped as track, dataset, or global rates. The
contact-drive node remains per track/dataset plate, so a shared driven-rate node
can still receive different contact probabilities from different traces.

If contact is defined by thresholding a score, pass the score array directly.
The model builds an unpinned `RcNode` which emits `p_contact(t)` from the
current threshold and updates `rc` from the promoter-state Markov blanket:

```python
dataset = MS2Dataset(
    observed=observed,
    noise_std=np.float32(0.5),
    dt=np.float32(0.5),
)

config = ModelConfig(
    n_states=2,
    driven_transition_indices=(1,),
    contact_drives=(contact_score.astype(np.float32),),
    rc_initial=np.float32(0.3),
    rc_bounds=(0.1, 1.0),
    rc_candidate_values=np.linspace(0.1, 1.0, 10, dtype=np.float32),
)

model = ViprodyneModel(datasets=(dataset,), config=config)
```

When `rc_candidate_values` is omitted, the model derives a candidate grid from
the finite score values inside `rc_bounds`. Explicit candidates are preferred
when the user wants a specific profile grid or comparable output across fits.

## Contact-Threshold Profiles

Notebook-style `rc` profiles, where an external score is thresholded into a
contact probability, are supported by `profile_contact_threshold(...)`:

```python
from viprodyne import CAVIConfig, profile_contact_threshold

config = ModelConfig(
    n_states=2,
    driven_transition_indices=(1,),
    contact_drives=(contact_score.astype(np.float32),),
    ms2_kernel="proximal",
)
fit_config = CAVIConfig(max_iterations=100)

profile = profile_contact_threshold(
    datasets=(dataset,),
    config=config,
    candidate_values=np.linspace(0.25, 2.0, 10, dtype=np.float32),
    fit_config=fit_config,
)

best_fit = profile.best_fit
best_threshold = profile.best_value
```

This helper creates a fresh `ViprodyneModel` for each candidate using the same
config-level drive pathway as MAP `rc` fitting, runs `model.run_inference(...)`,
and returns the ELBO profile. It is intended as the package-native version of
the ad hoc profile loop used in the
`1_kon_rc_toy_identify.ipynb` notebook. A rendered package example lives at
`examples/contact_threshold_profile.ipynb`, with the same workflow mirrored in
`examples/contact_threshold_profile.py`.

## Current Gaps

- Dense mean-field/exact Pol2 modes remain available internally, but should not
  be used for large regular MS2 traces because dense memory scales as
  `O(n_observations * n_loadings)`.
- Direct `RcNode` threshold updates are MAP/grid updates. Full posterior
  quadrature over `rc` is still pending.

## Verification Strategy

Tests cover:

- transition indexing and column-sum-zero generator wrapping;
- CTMC forward solutions, expected occupancy, and expected jumps against
  analytic two-state checks;
- missing-data masking and finite gradients;
- exact Pol2 enumeration, transfer likelihood, transfer marginals, and entropy
  against exhaustive enumeration;
- continuous Pol2 sampler setup, missing-data sampler smoke tests, and
  thermodynamic-integration log-partition smoke tests;
- non-interacting Pol2 theory when the kernel is shorter than the sampling
  interval;
- large Pol2 batches with 200 tracks and 1000 timepoints;
- synthetic contact-threshold recovery against latent Pol2 loadings, predicted
  signal, and promoter state probabilities;
- Gamma/Dirichlet parameter entropy and pinned-node behavior;
- driven contact-survival MAP profiles and driven promoter tilts;
- contact-threshold profile plumbing and direct in-graph `RcNode` threshold
  updates;
- graph Markov-blanket wiring;
- top-level graph construction with shared, scoped, and driven parameter nodes;
- CAVI convergence monitoring without per-iteration ELBO evaluation.
