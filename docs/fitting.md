# Fitting

The main fitting entry point is
{meth}`viprodyne.ViprodyneModel.run_inference`.

```python
fit = model.run_inference(max_iterations=100, tolerance=1e-4, progress=True)
```

This runs coordinate-ascent variational inference. Convergence is monitored
from parameter changes, and the ELBO is computed after the final sweep. The
`progress` option prints the iteration time, elapsed time, ETA, and largest
unconverged parameter nodes during fitting.

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

Promoter-state and Pol2-loading posteriors are reported on the same latent
grid. If `latent_grid` is omitted, the latent grid defaults to the dataset
measurement times. Set {attr}`viprodyne.ModelConfig.latent_grid` when you want
the inference itself to run on a different latent grid:

```python
latent_grid = np.arange(0.0, 600.0, 5.0, dtype=np.float32)

config = ModelConfig(
    n_states=2,
    pol2_mode="sampler",
    latent_grid=latent_grid,
)

model = ViprodyneModel(datasets=(dataset,), config=config)
fit = model.run_inference(fit_config)

condition = fit.datasets["dataset_0"]
state_posterior = condition.state_posterior
loading_posterior = condition.loading_posterior
predicted_signal = condition.predicted_signal
```

The latent grid is shared by the promoter-state path and Pol2 loading
variables, so no posterior resampling is needed between those nodes. Signal
predictions remain on the dataset observation times. For multiple datasets,
`latent_grid` can be a tuple in dataset order or a dictionary keyed by dataset
name.

The transfer backend is the exception: it is only supported on the observation
latent grid. Passing `latent_grid` with `pol2_mode="transfer"` raises an error;
use `pol2_mode="sampler"` for fine latent grids.

Common {class}`viprodyne.DatasetInferenceResult` fields:

- `latent_grid`: latent times shared by promoter-state and Pol2-loading posteriors.
- `state_posterior`: promoter-state probabilities on `latent_grid`.
- `loading_posterior`: Pol2 loading probabilities on `latent_grid`.
- `loading_posterior_rate`: posterior loading rate for sampler mode.
- `predicted_signal`: posterior mean MS2 signal at the observation times.
- `transition_rates`: fitted promoter transition rates by transition index.
- `loading_rates`: fitted loading rates by promoter state.
- `contact_rc` and `contact_probability`: contact-drive outputs when configured.

## Pol2 Backends And ELBOs

For high-temporal-resolution data where the transfer backend is too large, use
`pol2_mode="sampler"` and set `latent_grid` to the grid you want for both
promoter states and Pol2 loadings. Signal predictions remain on the dataset
observation times.

```python
latent_grid = np.arange(0.0, 600.0, 5.0, dtype=np.float32)

config = ModelConfig(
    n_states=2,
    pol2_mode="sampler",
    latent_grid=latent_grid,
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
    latent_grid=latent_grid,
)
```

This does not change the sampler posterior or its messages to the rest of the
graph. It only changes the local Pol2 term reported in the final ELBO to the
mean-field diagnostic value.
