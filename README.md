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
