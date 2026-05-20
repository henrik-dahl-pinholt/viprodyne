# viprodyne

Fresh-start variational inference tools for MS2 posterior models.

The package is organized in layers:

- `viprodyne.core`: mathematical kernels such as transition-edge indexing and
  driven-rate contact-survival objectives.
- `viprodyne.variational`: reusable variational node contracts, conjugate
  parameter nodes, deterministic nodes, and graph/message plumbing.

CTMC generators use the column-sum-zero convention throughout:
`Q[to_state, from_state]` is the transition rate from `from_state` to
`to_state`, and probabilities evolve as `p(t + dt) = exp(Q dt) @ p(t)`.

Install locally for development:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```
