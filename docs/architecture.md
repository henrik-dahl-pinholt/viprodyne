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

`viprodyne.core.bernoulli_transfer_pol2` contains the Pol2 loading kernels:

- `mean_field_bernoulli_elbo_terms` and `mean_field_bernoulli_elbo` implement
  the legacy MS2Posterior ELBO term convention.
- `exact_bernoulli_posterior` enumerates all binary loading configurations for
  small systems and tests.
- `bernoulli_transfer_log_likelihood` computes exact loading log evidence with
  a sliding-window transfer algorithm.
- `bernoulli_transfer_posterior` computes exact transfer marginals and entropy
  from `log_Z` derivative identities. It avoids a materialized joint posterior,
  but it is more expensive than the log-likelihood-only transfer pass.

`viprodyne.core.pol2_sampler` contains the continuous-time reversible-jump Pol2
loading sampler and thermodynamic-integration log-partition estimator. This is
the sampler-style backend for continuous loading events rather than binary
loading variables on a fixed observation grid.

The dense `design_matrix` representation is the linear map from loading
variables to expected MS2 intensity, `I_mean = design_matrix @ tau`. It is useful
for tiny exact enumeration tests and generic mean-field checks, but it scales as
`O(n_observations * n_loadings)`. For regular MS2 kernels, prefer the transfer
representation (`window_weights` plus `observation_starts`) or the sampler,
which keep memory tied to kernel support and grid size instead of a full dense
convolution matrix.

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
  `transfer` modes.

Pinned parameter nodes emit deterministic moments and have zero entropy.

`PolymeraseLoadings` does not take a dataset-level Pol2 prior in model-built
graphs. Its Bernoulli loading prior is derived during updates from promoter
interval state probabilities and loading-rate moments:

```text
P(tau_i = 1) = sum_s q_i(s) * (1 - exp(-E[r_s] * dt_i))
```

The initial value inside the node is only a placeholder until graph messages
from `PromoterState` and `LoadingRate` are available.

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

Create one `MS2Dataset` per dataset plate and pass them to `ViprodyneModel`
with a `ModelConfig`.

```python
import numpy as np

from viprodyne import MS2Dataset, ModelConfig, ViprodyneModel

dataset = MS2Dataset(
    name="condition_0",
    observed=np.array([0.1, np.nan, 0.8], dtype=np.float32),
    noise_std=np.float32(0.5),
    design_matrix=np.eye(3, dtype=np.float32),
)

model = ViprodyneModel(
    datasets=(dataset,),
    config=ModelConfig(
        n_states=2,
        time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        shared_transition_rates=False,
        shared_loading_rates=False,
    ),
)

model.run_schedule(model.default_schedule())
posterior_state = model.graph.moments.get("condition_0:s")["posterior"]
```

Datasets can either inherit `ModelConfig.time_grid` or provide their own
`MS2Dataset.time_grid`. Per-dataset grids are the right representation when
conditions have different frame timing or different trace lengths.

Driven transitions are selected by transition index. For a two-state model,
index `1` is `0 -> 1`.

```python
dataset = MS2Dataset(
    name="condition_0",
    observed=np.array([0.1, 0.8], dtype=np.float32),
    noise_std=np.float32(0.5),
    contact_probability=np.array([0.25, 0.75], dtype=np.float32),
)

model = ViprodyneModel(
    datasets=(dataset,),
    config=ModelConfig(
        n_states=2,
        time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        driven_transition_indices=(1,),
        driven_rate_initial=np.float32(0.8),
        driven_rate_bounds=(1e-4, 10.0),
    ),
)
```

Shared rate nodes are enabled with `shared_transition_rates=True` or
`shared_loading_rates=True`. Driven transition rates can be shared across
datasets while the contact-drive node remains per dataset.

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
