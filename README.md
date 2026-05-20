# viprodyne

Fresh-start variational inference tools for MS2 posterior models.

The package is organized in layers:

- `viprodyne.core`: mathematical kernels such as transition-edge indexing and
  driven-rate contact-survival objectives.
- `viprodyne.variational`: reusable variational node contracts, conjugate
  parameter nodes, deterministic nodes, and graph/message plumbing.

Install locally for development:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```
