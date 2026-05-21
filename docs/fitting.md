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

Common {class}`viprodyne.DatasetInferenceResult` fields:

- `state_posterior`: promoter-state probabilities on the dataset time grid.
- `loading_posterior`: Pol2 loading probabilities on loading intervals.
- `predicted_signal`: posterior mean MS2 signal.
- `transition_rates`: fitted promoter transition rates by transition index.
- `loading_rates`: fitted loading rates by promoter state.
- `contact_rc` and `contact_probability`: contact-drive outputs when configured.
