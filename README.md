# viprodyne

`viprodyne` is a Python/JAX package for variational inference on MS2-like
live-imaging transcription data.

It fits promoter-state dynamics, polymerase loading probabilities, loading
rates, transition rates, and optional contact-driven transition effects from
fluorescence traces.

## Install

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev,docs]"
```

## Quick Start

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

print(fit.cavi.converged, fit.cavi.elbo)
print(posterior.state_posterior.shape)
print(posterior.loading_posterior.shape)
```

`MS2Dataset.observed` must have shape `(n_traces, n_timepoints)`. A single trace
should be passed as `(1, n_timepoints)`.

## Common Tasks

Use the built-in proximal MS2 kernel:

```python
config = ModelConfig(
    n_states=2,
    ms2_kernel="proximal",
    t_rise=np.float32(0.25),
    t_plateau=np.float32(0.75),
    rna_intensity=np.float32(1.0),
)
```

Share rates globally, per dataset group, or per track:

```python
config = ModelConfig(
    n_states=2,
    transition_rate_scope="global",
    loading_rate_scope="dataset",
)
```

Fit a contact-threshold drive inside one model:

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
    ),
)

fit = model.run_inference(max_iterations=100)
print(fit.datasets["condition_0"].contact_rc)
```

Run an outer threshold profile:

```python
from viprodyne import CAVIConfig, profile_contact_threshold

profile = profile_contact_threshold(
    datasets=(dataset,),
    config=ModelConfig(n_states=2, driven_transition_indices=(1,)),
    contact_scores=contact_score,
    candidate_values=np.linspace(0.1, 1.0, 10, dtype=np.float32),
    fit_config=CAVIConfig(max_iterations=100),
)

print(profile.best_value, profile.elbos)
```

## Examples

A rendered notebook with visible output is available at
[`examples/contact_threshold_profile.ipynb`](examples/contact_threshold_profile.ipynb).
The same workflow is also available as a script:

```bash
python examples/contact_threshold_profile.py
```

## Documentation

The Sphinx documentation source is in [`docs/`](docs/).

Build it locally with:

```bash
python -m pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The API reference is generated from docstrings with Sphinx autosummary.

## Development Checks

```bash
python -m ruff check .
python -m pytest
```
