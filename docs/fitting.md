# Fitting

The main fitting entry point is
{meth}`viprodyne.ViprodyneModel.run_inference`.

```python
fit = model.run_inference(max_iterations=100, tolerance=1e-4, progress=True)
```

This runs coordinate-ascent variational inference. Convergence is monitored
from parameter changes, and the ELBO is computed after the final sweep. The
`progress` option prints the largest unconverged parameter nodes during fitting.

Use {class}`viprodyne.CAVIConfig` when you want explicit control:

```python
from viprodyne import CAVIConfig

fit_config = CAVIConfig(
    max_iterations=200,
    min_iterations=5,
    tolerance=1e-5,
    progress=True,
    compute_elbo=True,
)

fit = model.run_inference(fit_config)
```

Posterior outputs can be requested on a separate grid without changing the
observation grid used for the MS2 signal prediction:

```python
posterior_times = np.arange(0.0, 600.0, 5.0, dtype=np.float32)
fit = model.run_inference(
    fit_config,
    posterior_times=posterior_times,
)

condition = fit.datasets["dataset_0"]
state_posterior = condition.state_posterior
loading_posterior = condition.loading_posterior
predicted_signal = condition.predicted_signal
```

`posterior_times` is applied to both promoter-state and Pol2-loading posterior
outputs. Use `state_times=` or `loading_times=` when those grids should differ.
For multiple datasets, pass a dictionary keyed by dataset name.

Common {class}`viprodyne.DatasetInferenceResult` fields:

- `state_posterior`: promoter-state probabilities on `state_posterior_times`.
- `loading_posterior`: Pol2 loading probabilities on `loading_posterior_times`.
- `loading_posterior_rate`: posterior loading rate for sampler mode.
- `predicted_signal`: posterior mean MS2 signal at the observation times.
- `transition_rates`: fitted promoter transition rates by transition index.
- `loading_rates`: fitted loading rates by promoter state.
- `contact_rc` and `contact_probability`: contact-drive outputs when configured.

## Pol2 Backends And ELBOs

For high-temporal-resolution data where the transfer backend is too large, use
`pol2_mode="sampler"` and set `sampler_fine_grid` to the Pol2 loading grid you
want to inspect. Signal predictions remain on the dataset observation times.

```python
config = ModelConfig(
    n_states=2,
    pol2_mode="sampler",
    sampler_fine_grid=np.arange(0.0, 600.0, 5.0, dtype=np.float32),
    sampler_iterations=20_000,
    sampler_repeats=50,
)
```

You can keep sampler posteriors while using a mean-field local Pol2 ELBO
diagnostic by setting `pol2_elbo_mode="mean_field"`:

```python
config = ModelConfig(
    n_states=2,
    pol2_mode="sampler",
    pol2_elbo_mode="mean_field",
    sampler_fine_grid=np.arange(0.0, 600.0, 5.0, dtype=np.float32),
)
```

This does not change the sampler posterior or its messages to the rest of the
graph. It only changes the local Pol2 term reported in the final ELBO to the
mean-field diagnostic value.
