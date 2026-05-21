# Getting Started

Install the package in editable mode during development:

```bash
python -m pip install -e .
```

For tests and docs:

```bash
python -m pip install -e ".[dev,docs]"
```

Minimal fit with {class}`viprodyne.MS2Dataset`,
{class}`viprodyne.ModelConfig`, and {class}`viprodyne.ViprodyneModel`:

```python
import numpy as np

from viprodyne import MS2Dataset, ModelConfig, ViprodyneModel

dataset = MS2Dataset(
    observed=np.array([[0.1, np.nan, 0.8]], dtype=np.float32),
    noise_std=np.float32(0.5),
    dt=np.float32(0.5),
)

config = ModelConfig(
    n_states=2,
    ms2_kernel="proximal",
    t_rise=np.float32(0.25),
    t_plateau=np.float32(0.75),
    rna_intensity=np.float32(1.0),
)

model = ViprodyneModel(datasets=(dataset,), config=config)
fit = model.run_inference(max_iterations=100, tolerance=1e-4)
posterior = fit.datasets["dataset_0"]

print(fit.cavi.converged, fit.cavi.elbo)
print(posterior.state_posterior.shape)
print(posterior.loading_posterior.shape)
```

The returned {class}`viprodyne.DatasetInferenceResult` contains posterior state
probabilities, Pol2 loading probabilities, predicted signal, fitted transition
rates, fitted loading rates, and contact-drive outputs when present.
