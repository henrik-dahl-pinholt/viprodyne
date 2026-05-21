# Driven Contacts

Driven transitions let an external time-varying signal modulate a promoter
transition rate. In a two-state model, transition index `1` is `0 -> 1`.

If contact probabilities are known, keep the observations in `MS2Dataset` and
pass the drive when constructing the model:

```python
from viprodyne import ContactDrive

dataset = MS2Dataset(
    name="condition_0",
    observed=observed,
    noise_std=np.float32(0.5),
    time_grid=time_grid,
)

config = ModelConfig(
    n_states=2,
    driven_transition_indices=(1,),
    driven_rate_initial=np.float32(0.8),
    driven_rate_bounds=(1e-4, 10.0),
)

model = ViprodyneModel(
    datasets=(dataset,),
    config=config,
    contact_drive=ContactDrive.fixed(contact_probability),
)
```

If contact is defined by thresholding a score, pass a threshold contact drive
and let `RcNode` update the threshold inside the model:

```python
dataset = MS2Dataset(
    name="condition_0",
    observed=observed,
    noise_std=np.float32(0.5),
    time_grid=time_grid,
)

config = ModelConfig(
    n_states=2,
    driven_transition_indices=(1,),
    rc_initial=np.float32(0.3),
    rc_bounds=(0.1, 1.0),
    rc_candidate_values=np.linspace(0.1, 1.0, 10, dtype=np.float32),
)

model = ViprodyneModel(
    datasets=(dataset,),
    config=config,
    contact_drive=ContactDrive.threshold(contact_score),
)
```

For a general rc-dependent drive, pass a function returning contact
probabilities on intervals or grid points:

```python
def p_contact_from_rc(time_grid, rc):
    return ep_posterior_probability(time_grid[:-1], rc)

model = ViprodyneModel(
    datasets=(dataset,),
    config=config,
    contact_drive=ContactDrive.function(p_contact_from_rc),
)
```

If `rc_candidate_values` is omitted, the model derives a threshold grid from
the finite score values inside `rc_bounds`.

For a full outer profile, use `profile_contact_threshold`. It fits one model
per candidate threshold and returns the ELBO profile.
