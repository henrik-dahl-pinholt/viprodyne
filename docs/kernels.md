# MS2 Kernels

The built-in kernel is {class}`viprodyne.ProximalKernel`, a ramp-then-plateau
response specified by `t_rise`, `t_plateau`, and `rna_intensity`.

```python
config = ModelConfig(
    n_states=2,
    ms2_kernel="proximal",
    t_rise=np.float32(0.25),
    t_plateau=np.float32(0.75),
    rna_intensity=np.float32(1.0),
)
```

You can also pass a JAX-compatible callable:

```python
import jax.numpy as jnp


def rectangular_kernel(time_offsets):
    return jnp.where((time_offsets >= 0.0) & (time_offsets < 0.75), 1.0, 0.0)


config = ModelConfig(
    n_states=2,
    ms2_kernel=rectangular_kernel,
)
```

For regular MS2 traces, `pol2_mode="auto"` uses the transfer backend. This
avoids constructing a dense observation-by-loading matrix for long traces.
