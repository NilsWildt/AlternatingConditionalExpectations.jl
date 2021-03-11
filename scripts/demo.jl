using DrWatson
@quickactivate "AlternatingConditionalExpectation"
include(srcdir("utils.jl"))
include(srcdir("benchmark_functions.jl"))
using Statistics
using Pkg
using Test
Base.Experimental.@optlevel 3
using StatsPlots
# Pkg.add(url = "git@github.com:NilsWildt/AlternatingConditionalExpectation.jl.git")
using AlternatingConditionalExpectation
using Plots
using Random
ENV["JULIA_DEBUG"] = "all"

Nk = 2000
Benchmark =  f_b1(Nk, 1, "normal", 1.0, 1.0, false, 0, false, (-2., 2.))
# Benchmark =  f_b1(Nk, 1, "normal", 1.0, 0.2, false, 0, false, (-3., 3.))
# Benchmark =  f_b2(Nk, 1, "uniform", 1.0, 0.5, false, 0, false, (-3., 3.))
# Benchmark =  f_toy(Nk, 1, "normal", 1.0, 1.0, false, 0, scale_data, (-12., 12.))
X =  Benchmark.X
Y =  Benchmark.Y

# mysmoother1 = AlternatingConditionalExpectation.LAS5(Nk ÷ 5)
# mysmoother2 = AlternatingConditionalExpectation.LASb(Nk ÷ 6)
# mysmoother3 =  AlternatingConditionalExpectation.LLSS(Nk ÷ 3)
# mysmoother4 =  AlternatingConditionalExpectation.LLSSb(Nk ÷ 3)
mysmoother5 = AlternatingConditionalExpectation.FRSS([0.05,0.1,0.5], 0.2, 0.2)
# # σ = 0.2
# mykernel = AlternatingConditionalExpectation.Kernelregression.Gaussian(σ)
# reg = 1E-3
# mysmoother2 = AlternatingConditionalExpectation.NWKernelsmooth(mykernel)
mysmoothers = Array{Smoother}([mysmoother5])
# guess_parameters!(mysmoother, Nk)
myace =  AlternatingConditionalExpectation.Acerun(X, Y, mysmoothers, 1E-5, 20, 500)
Benchmark.Φ_x, Benchmark.Θ_y, Benchmark.sIx, Benchmark.sIy, Benchmark.conv_err =    ACE_multivariate(myace)
h1 =  benchmark_ace_plot(Benchmark, "ACE Demo", 80) 