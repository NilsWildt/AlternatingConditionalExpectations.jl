"""
    ACE

Alternating Conditional Expectations — a Julia implementation of the ACE
algorithm (Breiman & Friedman, 1985) for nonparametric regression.

ACE estimates optimal transformations `Φ(X)` and `Θ(Y)` such that
`Θ(Y) ≈ Σⱼ Φⱼ(Xⱼ)`, using an iterative backfitting loop where each conditional
expectation is estimated with a local smoother from
[`LocalSmoothers`](@ref).

# Example
```julia
using ACE
X, Y = ACE.generate_bivariate_data(Float64, 200, 0.1, 1.0, 42)
res = ace_run(ACEsim(X, Y, LASb(10)))
plot_ace_results(res; savepath = "ace_result.png")
```
"""
module ACE

using ConcreteStructs: @concrete
using DispatchDoctor
using Statistics
using Random
using Plots
using LaTeXStrings
using StaticArrays
using PrecompileTools

import LocalSmoothers
import LocalSmoothers: Smoother, do_smoothing,
    LAS, LASb, LLSS, LLSSb, FRSS, Kernelsmooth, NWKernelsmooth
import SimpleKernelRegression:
    Gaussian, Imq, Mq, Polynomial, Linear, Epanechnikov, Wendland

export ace, avas, predict, FitControls
export ace_run, avas_run, ctsub, ACEsim, ACEres, generate_bivariate_data, stoch_normalize, ε²
export VarTransform, Smooth, Monotone, LinearFit, Categorical, Periodic
export plot_ace_results, benchmark_ace_plot
export BenchmarkFunction, f_b1, f_b2, f_b3, f_b4, f_toy
export normal_sample, uniform_sample, get_sample
export MAE, nMAE, RMSE, UFV, myErr, pErr, AARD
# smoother API re-exported for convenience
export Smoother, do_smoothing, LAS, LASb, LLSS, LLSSb, FRSS, Kernelsmooth, NWKernelsmooth
export Gaussian, Imq, Mq, Polynomial, Linear, Epanechnikov, Wendland

include("results.jl")
include("utils.jl")
include("error_utils.jl")
include("algorithm.jl")
include("transforms.jl")
include("avas.jl")
include("fit.jl")
include("benchmarks.jl")
include("plotting.jl")

@compile_workload begin
    X, Y = generate_bivariate_data(Float64, 100, 0.1, 1.0, 42)
    res = ace_run(ACEsim(X, Y, LASb(8)))
    plot_ace_results(res)
end

end # module
