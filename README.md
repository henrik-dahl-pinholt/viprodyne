# viprodyne

Fresh-start variational inference tools for MS2 posterior models.

The package is organized in three layers:

- `viprodyne.core`: mathematical kernels such as transition-edge indexing and
  driven-rate contact-survival objectives, tilted CTMC solves, simulation, and
  JAX Pol2 loading kernels.
- `viprodyne.variational`: reusable variational node contracts, conjugate
  parameter nodes, domain nodes, and graph/message plumbing.
- `viprodyne.model`: the top-level dataset/config interface that constructs
  per-dataset graph plates with optional shared parameter nodes.

CTMC generators use the column-sum-zero convention throughout:
`Q[to_state, from_state]` is the transition rate from `from_state` to
`to_state`, and probabilities evolve as `p(t + dt) = exp(Q dt) @ p(t)`.
Transition-rate vectors are ordered by scanning off-diagonal matrix entries
row by row, e.g. a three-state model uses `[[-, 0, 1], [2, -, 3], [4, 5, -]]`.

The numerical kernels are JAX-first and keep arrays in `float32`. Do not enable
JAX x64 for normal development or tests.

The top-level model API derives Pol2 observation internals from an MS2 kernel.
Users pass observations, noise, timing, and either a named kernel or a custom
JAX-compatible kernel function:

```python
import numpy as np

from viprodyne import MS2Dataset, ModelConfig, ViprodyneModel

dataset = MS2Dataset(
    name="condition_0",
    observed=np.array([[0.1, np.nan, 0.8]], dtype=np.float32),
    noise_std=np.float32(0.5),
    time_grid=np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
)

model = ViprodyneModel(
    datasets=(dataset,),
    config=ModelConfig(
        n_states=2,
        ms2_kernel="proximal",
        t_rise=np.float32(0.25),
        t_plateau=np.float32(0.75),
        rna_intensity=np.float32(1.0),
    ),
)

fit = model.run_inference(max_iterations=100, tolerance=1e-4)
posterior = fit.datasets["condition_0"]

state_posterior = posterior.state_posterior
loading_posterior = posterior.loading_posterior
predicted_ms2 = posterior.predicted_signal
loading_rates = posterior.loading_rates
transition_rates = posterior.transition_rates

print(fit.cavi.converged, fit.cavi.elbo)
```

`MS2Dataset.observed` must have shape `(n_traces, n_timepoints)`, including
single-trace datasets as `(1, n_timepoints)`. Use `MS2Dataset.rate_group` plus
`transition_rate_scope` and `loading_rate_scope` to choose per-track,
per-dataset, or global rate nodes. The standard fitting entry point is
`model.run_inference(...)` or its alias `model.fit(...)`. It runs
coordinate-ascent VI, monitors convergence by parameter changes, computes the
ELBO only after the final sweep, and returns structured per-dataset outputs.
Promoter and Pol2 updates use natural expected-log messages, including
`E[log pi]` for initial state weights and expected log load/no-load terms for
Pol2 priors. Missing observations are also propagated to an internal Pol2
loading mask so prior-only intervals do not update promoter or loading-rate
factors.

`pol2_mode="auto"` uses the memory-efficient transfer backend. The continuous
Pol2 sampler is available with `pol2_mode="sampler"` for proximal MS2 kernels.

Notebook-style contact-threshold profiles can be run with
`profile_contact_threshold(...)`. This fits one model per candidate threshold
and returns the ELBO profile plus the best structured fit:

```python
from viprodyne import CAVIConfig, profile_contact_threshold

profile = profile_contact_threshold(
    datasets=(dataset,),
    config=ModelConfig(
        n_states=2,
        driven_transition_indices=(1,),
        ms2_kernel="proximal",
    ),
    contact_scores=contact_score,
    candidate_values=np.linspace(0.25, 2.0, 10, dtype=np.float32),
    fit_config=CAVIConfig(max_iterations=100),
)

print(profile.best_value, profile.elbos)
```

For coordinate updates inside one model, pass the score on the dataset and,
optionally, an `rc` candidate grid on the config:

```python
dataset = MS2Dataset(
    name="condition_0",
    observed=observed,
    noise_std=np.float32(0.5),
    time_grid=time_grid,
    contact_score=contact_score,
)

model = ViprodyneModel(
    datasets=(dataset,),
    config=ModelConfig(
        n_states=2,
        driven_transition_indices=(1,),
        rc_initial=np.float32(0.3),
        rc_bounds=(0.1, 1.0),
        rc_candidate_values=np.linspace(0.1, 1.0, 10, dtype=np.float32),
    ),
)

fit = model.run_inference(CAVIConfig(max_iterations=100))
print(fit.datasets["condition_0"].contact_rc)
```

If `rc_candidate_values` is omitted, viprodyne builds a threshold grid from the
finite contact-score values inside `rc_bounds`.

A rendered notebook with visible output is in
`examples/contact_threshold_profile.ipynb`; the same workflow is also available
as a small script in `examples/contact_threshold_profile.py`.

Install locally for development:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

Run lint checks:

```bash
python -m ruff check .
```

More detail on the current architecture and conventions is in
[`docs/architecture.md`](docs/architecture.md).
