# viprodyne

`viprodyne` is a Python/JAX package for fitting promoter-state and polymerase-loading
models to MS2-like live-imaging transcription data.

The package is built around one workflow:

1. Put one experiment or condition into an {class}`viprodyne.MS2Dataset`.
2. Choose a promoter model, MS2 kernel, and rate-sharing structure with
   {class}`viprodyne.ModelConfig`.
3. Run {meth}`viprodyne.ViprodyneModel.run_inference`.
4. Inspect state posteriors, loading posteriors, fitted rates, and predicted signal.

```{toctree}
:maxdepth: 2

getting_started
data
fitting
kernels
driven_contacts
rate_sharing
missing_data
examples
api/index
developer/architecture
developer/ms2posterior_parity
```
