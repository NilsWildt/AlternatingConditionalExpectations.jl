using DrWatson
@quickactivate "ACE"
include(srcdir("utils.jl"))
include(srcdir("benchmark_functions.jl"))
using Statistics
using Pkg
using Test
Base.Experimental.@optlevel 3
using StatsPlots
# Pkg.add(url = "git@github.com:NilsWildt/ACE.jl.git")
using ACE
using Plots
using Random
ENV["JULIA_DEBUG"] = "all"

Nk = 2000
# Benchmark =  f_b2(Nk, 1, "normal", 1.0, 1.0, false, 42, true, (-2., 2.))
Benchmark =  f_b1(Nk, 1, "uniform", 1.0, 1.0, false, 0, false, (-3., 3.))
# Benchmark =  f_b4(Nk, 1, "uniform", 1.0, 0.5, false, 0, false, (-5., 5.))
# Benchmark =  f_toy(Nk, 1, "normal", 1.0, 0.0, false, 0, false, (-12., 12.))
X =  Benchmark.X
Y =  Benchmark.Y

# mysmoother1 = ACE.LAS(Nk ÷ 2)
# mysmoother2 = ACE.LASb(Nk ÷ 5)
mysmoother3 =  ACE.LLSS(Nk ÷ 3)
mysmoother4 =  ACE.LLSSb(Nk ÷ 4)
mysmoother5 = ACE.FRSS([0.05,0.1,0.5], 0.2, 0.2)
# σ = 0.1
# mykernel = ACE.Kernelregression.Gaussian(σ)
# reg = 1.0
# mysmoother2 = ACE.NWKernelsmooth(mykernel)
mysmoothers = Array{Smoother}([mysmoother4])
# guess_parameters!(mysmoother, Nk)
myace =  ACE.ACEsim(X, Y, mysmoothers)
res  =    run(myace)
myres = ACE.ACEres(X,Y,res...)
# Benchmark.Φ_x, Benchmark.Θ_y, Benchmark.sIx, Benchmark.sIy, Benchmark.conv_err
# h1 =  benchmark_ace_plot(Benchmark, "ACE Demo", 80) 
