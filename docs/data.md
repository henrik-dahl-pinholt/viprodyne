# Data

Each dataset stores a group of traces that share timing, noise, kernel settings,
and graph structure.

```python
dataset = MS2Dataset(
    name="condition_0",
    observed=observed,
    noise_std=np.float32(0.5),
    time_grid=time_grid,
)
```

`observed` must have shape `(n_traces, n_timepoints)`. A single trace should be
passed as `(1, n_timepoints)`.

`name` is optional. If omitted, `ViprodyneModel` assigns deterministic names
such as `dataset_0`, skipping any explicit names already in use.

`time_grid` defines Pol2 loading intervals and has length `n_timepoints + 1`
for the common case where observations are made at the right edge of each
interval. Use `sampling_times` when image acquisition times differ from
`time_grid[1:]`. If `sampling_times` is provided without `time_grid`, viprodyne
infers interval boundaries from adjacent frame midpoints.

Missing observations can be encoded as `NaN` or with `finite_mask`. Missing
values are excluded from the imaging likelihood and from downstream loading-rate
sufficient statistics.
