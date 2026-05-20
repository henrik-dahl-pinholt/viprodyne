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
    observed=np.array([0.1, np.nan, 0.8], dtype=np.float32),
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
```

For multi-trace fits, pass one `MS2Dataset` per trace. Use
`MS2Dataset.rate_group` plus `transition_rate_scope` and `loading_rate_scope` to
choose per-track, per-dataset, or global rate nodes. Run coordinate-ascent VI
with `model.fit_cavi(...)`; convergence is monitored by parameter changes and
the ELBO is computed only after the final sweep.

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
