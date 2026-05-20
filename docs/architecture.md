# viprodyne Architecture

`viprodyne` is a fresh implementation of the MS2Posterior variational model.
The code is split into numerical kernels, variational nodes, and a top-level
model builder so the graphical structure lives above the node implementations.

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

- `mean_field_bernoulli_elbo_terms` and `mean_field_bernoulli_elbo` implement
  the legacy MS2Posterior ELBO term convention.
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

- `MS2Kernel` can select a named parameterized kernel or wrap a custom
  JAX-compatible kernel function.
- `ms2posterior_kernel` is the current built-in parameterized kernel. It uses
  `t_rise`, `t_plateau`, and `rna_intensity`.
- `build_ms2_observation_model` derives the dense or transfer representation
  used by `PolymeraseLoadings`. The transfer path extracts row-specific windows
  from the kernel without storing a full dense observation-by-loading matrix.

`viprodyne.core.pol2_sampler` contains the continuous-time reversible-jump Pol2
loading sampler and thermodynamic-integration log-partition estimator. The core
module is implemented and tested; it is not yet exposed as a
`PolymeraseLoadings` node mode in the graph builder.

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
interval state probabilities and loading-rate moments:

```text
P(tau_i = 1) = sum_s q_i(s) * (1 - exp(-E[r_s] * dt_i))
```

The initial value inside the node is only a placeholder until graph messages
from `PromoterState` and `LoadingRate` are available.

Current limitation: the `PolymeraseLoadings` graph node assumes one trajectory
plate at a time. Batched transfer likelihoods exist at the core-kernel level,
but batched/plated loading-rate priors and sampler-mode graph integration still
need explicit node-level interfaces.

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

Create one `MS2Dataset` per dataset plate and pass them to `ViprodyneModel` with
a `ModelConfig`.

`MS2Dataset` contains observed intensities, noise, optional per-dataset
`time_grid`, optional `sampling_times`, optional missing-data mask, and optional
contact probabilities for driven transitions. It does not require users to pass
dense design matrices or transfer windows. Those are derived internally from the
MS2 kernel in `ModelConfig`.

`MS2Dataset` does not contain `prior_load_probabilities`. In model-built graphs,
those are derived from `PromoterState` and `LoadingRate` parents.

```python
import numpy as np

from viprodyne import MS2Dataset, ModelConfig, ViprodyneModel

dataset = MS2Dataset(
    name="condition_0",
    observed=np.array([0.1, np.nan, 0.8], dtype=np.float32),
    noise_std=np.float32(0.5),
    time_grid=np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
)

model = ViprodyneModel(
    datasets=(dataset,),
    config=ModelConfig(
        n_states=2,
        shared_transition_rates=False,
        shared_loading_rates=False,
        pol2_mode="auto",
        ms2_kernel="ms2posterior",
        t_rise=np.float32(0.25),
        t_plateau=np.float32(0.75),
        rna_intensity=np.float32(1.0),
    ),
)

model.run_schedule(model.default_schedule())
state_posterior = model.graph.moments.get("condition_0:s")["posterior"]
loading_posterior = model.graph.moments.get("condition_0:tau")["load_probabilities"]
```

Datasets can either inherit `ModelConfig.time_grid` or provide their own
`MS2Dataset.time_grid`. Per-dataset grids are the right representation when
conditions have different frame timing or different trace lengths.

By default, observations are assumed to occur at interval ends,
`time_grid[1:len(observed) + 1]`. Use `MS2Dataset.sampling_times` when frame
times are not the interval ends.

`ModelConfig.ms2_kernel` can be a named kernel string, an `MS2Kernel` instance,
or a custom JAX-compatible callable. The current built-in parameterized kernel is
`"ms2posterior"` and uses `t_rise`, `t_plateau`, and `rna_intensity`.

`ModelConfig.pol2_mode="auto"` chooses the transfer backend from the kernel
configuration. `pol2_mode="mean_field"` or `"exact"` asks the builder to create
the internal dense representation and should be reserved for small tests and
diagnostics.

Example with heterogeneous grids:

```python
d0 = MS2Dataset(
    name="d0",
    observed=np.array([0.1, 0.8], dtype=np.float32),
    noise_std=np.float32(0.5),
    time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
)
d1 = MS2Dataset(
    name="d1",
    observed=np.array([0.2, 0.5, 0.9], dtype=np.float32),
    noise_std=np.float32(0.5),
    time_grid=np.array([0.0, 0.25, 0.75, 1.5], dtype=np.float32),
)
model = ViprodyneModel(datasets=(d0, d1), config=ModelConfig(n_states=2))
```

Example with a custom kernel:

```python
import jax.numpy as jnp

def rectangular_kernel(time_offsets):
    return jnp.where((time_offsets >= 0.0) & (time_offsets < 0.75), 1.0, 0.0)

model = ViprodyneModel(
    datasets=(dataset,),
    config=ModelConfig(n_states=2, ms2_kernel=rectangular_kernel),
)
```

Driven transitions are selected by transition index. For a two-state model,
index `1` is `0 -> 1`.

```python
dataset = MS2Dataset(
    name="condition_0",
    observed=np.array([0.1, 0.8], dtype=np.float32),
    noise_std=np.float32(0.5),
    time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
    contact_probability=np.array([0.25, 0.75], dtype=np.float32),
)

model = ViprodyneModel(
    datasets=(dataset,),
    config=ModelConfig(
        n_states=2,
        driven_transition_indices=(1,),
        driven_rate_initial=np.float32(0.8),
        driven_rate_bounds=(1e-4, 10.0),
    ),
)
```

Shared rate nodes are enabled with `shared_transition_rates=True` or
`shared_loading_rates=True`. Driven transition rates can be shared across
datasets while the contact-drive node remains per dataset.

## Current Gaps

- The continuous Pol2 sampler is implemented as a core kernel, but it still needs
  a `PolymeraseLoadings(mode="sampler")` adapter and model-builder inputs for
  `fine_grid`, sampler iterations, repeats, and thermodynamic-integration
  settings.
- Dense mean-field/exact Pol2 modes remain available internally, but should not
  be used for large regular MS2 traces because dense memory scales as
  `O(n_observations * n_loadings)`.
- Driven `RcNode` currently supports pinned contact probabilities or an injected
  objective function. Full profile-likelihood rc updates from GPR/contact data
  still need a production data adapter.

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
- Gamma/Dirichlet parameter entropy and pinned-node behavior;
- driven contact-survival MAP profiles and driven promoter tilts;
- top-level graph construction with shared and driven parameter nodes.
