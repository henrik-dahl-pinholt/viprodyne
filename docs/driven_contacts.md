# Driven Contacts

Driven transitions let an external time-varying signal modulate a promoter
transition rate. In a two-state model, transition index `1` is `0 -> 1`.

Contact-drive inputs live in {class}`viprodyne.ModelConfig`, next to
`driven_transition_indices`. Pass one entry per dataset, in the same order as
the datasets passed to {class}`viprodyne.ViprodyneModel`.

## Fixed Probabilities

If contact probabilities are already known, pass a function that ignores `rc`
and returns the probability vector:

```python
from viprodyne import MS2Dataset, ModelConfig, ViprodyneModel


def fixed_contact(rc):
    del rc
    return contact_probability.astype(np.float32)


dataset = MS2Dataset(
    observed=observed,
    noise_std=np.float32(0.5),
    dt=np.float32(0.5),
)

config = ModelConfig(
    n_states=2,
    driven_transition_indices=(1,),
    contact_drives=(fixed_contact,),
    driven_rate_initial=np.float32(0.8),
    driven_rate_bounds=(1e-4, 10.0),
)

model = ViprodyneModel(datasets=(dataset,), config=config)
fit = model.run_inference(max_iterations=100)
```

## Threshold Scores

If contact is defined by thresholding a score, pass the score array directly.
The model treats contact as `score < rc`, and the internal `rc` node updates the
threshold from the promoter-state Markov blanket.

```python
from viprodyne import MS2Dataset, ModelConfig, ViprodyneModel

dataset = MS2Dataset(
    observed=observed,
    noise_std=np.float32(0.5),
    dt=np.float32(0.5),
)

config = ModelConfig(
    n_states=2,
    driven_transition_indices=(1,),
    contact_drives=(contact_score.astype(np.float32),),
    rc_initial=np.float32(0.3),
    rc_bounds=(0.1, 1.0),
    rc_candidate_values=np.linspace(0.1, 1.0, 10, dtype=np.float32),
)

model = ViprodyneModel(datasets=(dataset,), config=config)
fit = model.run_inference(max_iterations=100)
```

If `rc_candidate_values` is omitted, viprodyne derives a candidate grid from the
finite score values inside `rc_bounds`.

## Probability Functions

For a general `rc`-dependent drive, pass a callable. It may be `fn(rc)` or
`fn(times, rc)` and should return contact probabilities on the model intervals.

```python
def p_contact_from_rc(rc):
    return ep_posterior_probability(rc).astype(np.float32)


config = ModelConfig(
    n_states=2,
    driven_transition_indices=(1,),
    contact_drives=(p_contact_from_rc,),
    rc_initial=np.float32(0.5),
    rc_bounds=(0.1, 1.0),
)

model = ViprodyneModel(datasets=(dataset,), config=config)
```

## Contact-Threshold Profiles

Use {func}`viprodyne.profile_contact_threshold` for an outer ELBO profile. It
fits one model per threshold candidate using the same config-level contact-drive
pathway as MAP `rc` fitting.

```python
from viprodyne import CAVIConfig, profile_contact_threshold

candidate_values = np.array([0.25, 0.5, 0.75], dtype=np.float32)

config = ModelConfig(
    n_states=2,
    driven_transition_indices=(1,),
    contact_drives=(contact_score.astype(np.float32),),
)
fit_config = CAVIConfig(max_iterations=100)

profile = profile_contact_threshold(
    datasets=(dataset,),
    config=config,
    candidate_values=candidate_values,
    fit_config=fit_config,
)
```
