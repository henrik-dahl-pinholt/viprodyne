# Legacy Parity Investigation

This note records the current comparison against the legacy `MS2Posterior`
implementation. It is a developer migration check, not part of the public model
API.

Run it from the repository root:

```bash
python scripts/compare_ms2posterior_viprodyne.py
```

Use `--ms2posterior-path` if the legacy checkout is not at
`/net/levsha/share/hdp/git_reps/MS2Posterior`.

## What Is Held Fixed

The script builds a two-state, two-trace toy problem with the proximal kernel
support equal to one grid interval. It then compares:

- the raw Bernoulli transfer kernels with identical prior probabilities and
  transfer windows;
- the Pol2 node update with native Viprodyne priors and with legacy priors
  forced into Viprodyne;
- the promoter update with the legacy Pol2 message converted into Viprodyne's
  interval-count convention;
- Gamma rate-node updates from identical sufficient statistics;
- grouped ELBO terms after synchronizing the local node states.

## Current Findings

With identical Bernoulli transfer inputs, the Pol2 transfer kernels agree to
float32 precision:

```text
log Z                           max_abs=9.53674e-07
posterior loading probabilities max_abs=1.19209e-07
predicted MS2                   max_abs=1.19209e-07
```

The native Pol2 node updates do not produce identical priors:

```text
old prior vs viprodyne native prior       max_abs=0.0056106
old posterior vs viprodyne native posterior max_abs=0.00325286
old log Z vs viprodyne native log Z       max_abs=0.081809
```

This is expected. The legacy code plugs `exp(E[log k])` into
`1 - exp(-k dt)`. Viprodyne uses the Bernoulli CAVI log factors,
`E[log(1 - exp(-k dt))]` and `-E[k] dt`, when loading rates are Gamma
distributed. When the legacy prior probabilities are forced into Viprodyne, the
Pol2 posterior and log partition agree to float32 precision.

The promoter state posterior agrees when Viprodyne receives the legacy-scaled
Pol2 count message:

```text
state posterior max_abs=1.78814e-07
```

The promoter sufficient statistics and promoter ELBO term still differ:

```text
transition counts          max_abs=0.0439439
state exposure             max_abs=0.106803
promoter ELBO contribution max_abs=0.0173631
```

The reason is that legacy MS2Posterior uses grid-density/Riemann sufficient
statistics from the tilted CTMC posterior. Viprodyne uses interval-integrated
CTMC sufficient statistics from Frechet derivatives of the interval transition
operators.

Given identical sufficient statistics, the Gamma rate-node updates agree to
float32 precision:

```text
loading shape     max_abs=0
loading rate      max_abs=4.76837e-07
transition shape  max_abs=2.38419e-07
transition rate   max_abs=0
```

For transfer Pol2 loadings, the legacy loading-rate update treats posterior
loading probabilities like rate densities and multiplies counts by `dt`.
Viprodyne treats the same values as Bernoulli expected counts. In the toy case,
the largest difference between those two count conventions is:

```text
legacy dt-scaled loading counts vs Bernoulli counts max_abs=1.60416
```

After forcing the shared pieces to agree, the remaining total ELBO difference is
the promoter contribution:

```text
Loading Rates       diff=-1.25e-06
Polymerase Loadings diff= 4.77e-07
Promoter State      diff=-0.017363
Transition Rates    diff= 3.58e-07
total               diff=-0.017364
```

## Interpretation

The main discrepancy is not the transfer calculation itself. The raw transfer
kernel matches when supplied the same Bernoulli priors and windows.

The native CAVI path differs for two model-level reasons:

1. Viprodyne uses the Bernoulli loading CAVI log factors for Gamma-distributed
   loading rates, while legacy MS2Posterior uses a plug-in rate
   `exp(E[log k])`.
2. Viprodyne treats transfer load posteriors as Bernoulli expected counts,
   while legacy MS2Posterior passes them downstream as a rate-density-like
   message and multiplies by `dt`.

The remaining promoter ELBO/statistic difference comes from CTMC sufficient
statistics: interval-integrated in Viprodyne versus grid-density/Riemann in the
legacy code.
