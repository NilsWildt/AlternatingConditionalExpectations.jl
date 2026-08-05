# AlternatingConditionalExpectations.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![CI](https://github.com/NilsWildt/AlternatingConditionalExpectations.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/NilsWildt/AlternatingConditionalExpectations.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/NilsWildt/AlternatingConditionalExpectations.jl/coverage.svg?branch=master)](https://codecov.io/github/NilsWildt/AlternatingConditionalExpectations.jl?branch=master)

A Julia implementation of **Alternating Conditional Expectations** (ACE;
Breiman & Friedman, 1985) for nonparametric regression. ACE estimates optimal
transformations `Φⱼ(Xⱼ)` and `Θ(Y)` such that

```
Θ(Y) ≈ Σⱼ Φⱼ(Xⱼ)
```

by an iterative backfitting loop in which each conditional expectation is
estimated with a local smoother from
[LocalSmoothers.jl](https://github.com/NilsWildt/LocalSmoothers.jl).

## Installation

ACE depends on two unregistered packages, so add them first:

```julia
using Pkg
Pkg.add(url = "https://github.com/NilsWildt/SimpleKernelregression.jl")
Pkg.add(url = "https://github.com/NilsWildt/LocalSmoothers.jl")
Pkg.add(url = "https://github.com/NilsWildt/AlternatingConditionalExpectations.jl")
```

## Usage

```julia
using AlternatingConditionalExpectations

# 200 samples of a noisy bivariate relationship
X, Y = generate_bivariate_data(Float64, 200, 0.1, 1.0, 42)

# fit ACE with a boundary-corrected local-average smoother (window 10)
model = ace(X, Y; smoother = LASb(10))
model.tx        # φ, the predictor transforms
model.ty        # θ, the response transform
model.rsq       # coefficient of determination
predict(model, X)                     # transformed-response fit for new data

# AVAS (Tibshirani 1988): variance-stabilizing variant, monotone response
avas(X, Y; smoother = LASb(10))

# constrain individual variables
ace(X, Y; smoother = LASb(10), xtransforms = Monotone(LASb(10)))

# dark-theme diagnostic grid + convergence history
plot_ace_results(model; savepath = "ace_result.png")
```

`ace_run(ACEsim(...))` and `avas_run(...)` remain as thin compatibility shims.

Per-variable transforms: `Smooth` (default), `Monotone` (isotonic),
`LinearFit`, `Categorical`, and `Periodic`.

## What's in the package

- **Algorithms** — `ace` / `avas` (backfitting), `predict`, `stoch_normalize`, `ε²`, conditional
  expectations, bivariate/multivariate data generation.
- **Smoothers** — re-exported from `LocalSmoothers`: `LAS`, `LASb`, `LLSS`,
  `LLSSb`, `FRSS`, `Kernelsmooth`, `NWKernelsmooth`, plus the kernel types.
- **Benchmarks** — `BenchmarkFunction`, `f_b1`…`f_b4`, `f_toy` with samplers.
- **Error metrics** — `MAE`, `nMAE`, `RMSE`, `UFV`, `pErr`, `AARD`.
- **Plotting** — a `Plots.jl` recipe for `ACEres` plus `plot_ace_results` and
  `benchmark_ace_plot`.

## Development

```bash
git clone https://github.com/NilsWildt/AlternatingConditionalExpectations.jl
cd AlternatingConditionalExpectations.jl
julia --project=. -e 'import Pkg; Pkg.test()'
```

Runnable demos live in `scripts/` and write PNGs to `output/`:

```bash
julia --project=. scripts/demo2d.jl
```

## License

MIT — see [LICENSE.md](LICENSE.md).
