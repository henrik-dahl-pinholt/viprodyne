# Driven Contacts

Driven transitions let an external time-varying signal modulate a promoter
transition rate. In a two-state model, transition index `1` is `0 -> 1`.

If contact probabilities are known:

```python
dataset = MS2Dataset(
    name="condition_0",
    observed=observed,
    noise_std=np.float32(0.5),
    time_grid=time_grid,
    contact_probability=contact_probability,
)

config = ModelConfig(
    n_states=2,
    driven_transition_indices=(1,),
    driven_rate_initial=np.float32(0.8),
    driven_rate_bounds=(1e-4, 10.0),
)
```

If contact is defined by thresholding a score, pass `contact_score` and let
`RcNode` update the threshold inside the model:

```python
dataset = MS2Dataset(
    name="condition_0",
    observed=observed,
    noise_std=np.float32(0.5),
    time_grid=time_grid,
    contact_score=contact_score,
)

config = ModelConfig(
    n_states=2,
    driven_transition_indices=(1,),
    rc_initial=np.float32(0.3),
    rc_bounds=(0.1, 1.0),
    rc_candidate_values=np.linspace(0.1, 1.0, 10, dtype=np.float32),
)
```

If `rc_candidate_values` is omitted, the model derives a threshold grid from
the finite score values inside `rc_bounds`.

For a full outer profile, use `profile_contact_threshold`. It fits one model
per candidate threshold and returns the ELBO profile.
