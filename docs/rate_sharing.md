# Rate Sharing

Rates can be fitted per track, per dataset group, or globally.

```python
from viprodyne import ModelConfig

config = ModelConfig(
    n_states=2,
    transition_rate_scope="global",
    loading_rate_scope="dataset",
)
```

Use {attr}`viprodyne.MS2Dataset.rate_group` to share dataset-scoped rates
across several datasets:

```python
from viprodyne import MS2Dataset

dataset = MS2Dataset(
    name="replicate_0",
    rate_group="condition_a",
    observed=observed,
    noise_std=np.float32(0.5),
    dt=np.float32(0.5),
)
```

Use per-rate overrides when only selected rates should be shared:

```python
config = ModelConfig(
    n_states=2,
    transition_rate_scope="dataset",
    transition_rate_scopes={1: "global"},
    loading_rate_scope="dataset",
    loading_rate_scopes={0: "track"},
)
```
