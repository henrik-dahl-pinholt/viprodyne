# Missing Data

Missing MS2 observations can be represented directly as `NaN` values:

```python
observed = np.array([[0.2, np.nan, 0.8]], dtype=np.float32)
```

or by passing {attr}`viprodyne.MS2Dataset.finite_mask` with the same shape as
`observed`.

Missing observations are excluded from the imaging likelihood. The Pol2 loading
node also builds an internal loading mask, so intervals with no observational
support do not update promoter or loading-rate factors through the data.
