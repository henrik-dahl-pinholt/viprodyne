# Data

Each {class}`viprodyne.MS2Dataset` stores a group of traces that share timing,
noise, kernel settings, and graph structure. For regularly sampled data, pass
the frame spacing with `dt`; viprodyne centers each observation in its frame and
builds the internal loading intervals.

```python
dataset = MS2Dataset(
    observed=observed,
    noise_std=np.float32(0.5),
    dt=np.float32(0.5),
)
```

`observed` must have shape `(n_traces, n_timepoints)`. A single trace should be
passed as `(1, n_timepoints)`.

`name` is optional. If omitted, {class}`viprodyne.ViprodyneModel` assigns
deterministic names such as `dataset_0`, skipping any explicit names already in
use.

Use `sampling_times` instead of `dt` when image acquisition times are irregular:

```python
dataset = MS2Dataset(
    observed=observed,
    noise_std=np.float32(0.5),
    sampling_times=sampling_times.astype(np.float32),
)
```

Missing observations can be encoded as `NaN` or with `finite_mask`. Missing
values are excluded from the imaging likelihood and from downstream loading-rate
sufficient statistics.
