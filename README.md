# ACE.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![CI](https://github.com/NilsWildt/ACE.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/NilsWildt/ACE.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/NilsWildt/ACE.jl/coverage.svg?branch=master)](https://codecov.io/github/NilsWildt/ACE.jl?branch=master)

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
Pkg.add(url = "https://github.com/NilsWildt/ACE.jl")
```

## Usage

```julia
using ACE

# 200 samples of a noisy bivariate relationship
X, Y = ACE.generate_bivariate_data(Float64, 200, 0.1, 1.0, 42)

# fit ACE with a boundary-corrected local-average smoother (window 10)
res = ace_run(ACEsim(X, Y, LASb(10)))

# dark-theme diagnostic grid + convergence history
plot_ace_results(res; savepath = "ace_result.png")
```

## What's in the package

- **Algorithm** — `ace_run` (backfitting), `stoch_normalize`, `ε²`, conditional
  expectations, bivariate/multivariate data generation.
- **Smoothers** — re-exported from `LocalSmoothers`: `LAS`, `LASb`, `LLSS`,
  `LLSSb`, `FRSS`, `Kernelsmooth`, `NWKernelsmooth`, plus the kernel types.
- **Benchmarks** — `BenchmarkFunction`, `f_b1`…`f_b4`, `f_toy` with samplers.
- **Error metrics** — `MAE`, `nMAE`, `RMSE`, `UFV`, `pErr`, `AARD`.
- **Plotting** — a `Plots.jl` recipe for `ACEres` plus `plot_ace_results` and
  `benchmark_ace_plot`.

## Development

```bash
git clone https://github.com/NilsWildt/ACE.jl
cd ACE.jl
julia --project=. -e 'import Pkg; Pkg.test()'
```

Runnable demos live in `scripts/` and write PNGs to `output/`:

```bash
julia --project=. scripts/demo2d.jl
```

## License

MIT — see [LICENSE.md](LICENSE.md).
