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
using .ACE
using TimerOutputs
using Plots
using Random
ENV["JULIA_DEBUG"] = "all"
Nk = 500
# Benchmark =  f_b4(Nk, 1, "normal", 1.0, 1.0, false, 42, true, (-2., 2.))
# Benchmark2 =  f_b1(Nk, 1, "uniform", 1.0, 1.0, false, 0, false, (-3., 3.))
# Benchmark =  f_b2(Nk, 1, "uniform", 1.0, 0.5, false, 0, false, (-5., 5.))
# # Benchmark =  f_toy(Nk, 1, "normal", 1.0, 0.0, false, 0, false, (-12., 12.))
# m =  Benchmark.X
# X = hcat(m,Benchmark2.X)
rng =  MersenneTwister(42)    
 μ = 0.0
X1 =      randn(rng,  Float64, (Nk,1)) .+ μ
lb = -3
ub = 3.0
X2 = abs(ub - lb) .* (rand(rng,  Float64, (Nk,1))) .+ lb
lb = 0
ub = 0.5
X3 = abs(ub - lb) .* (rand(rng,  Float64, (Nk,1))) .+ lb
X = hcat(X3,X1,X3)
Y =  @. sin.(2*X1) + X2 + log.(X3)  
Y = Y .+ 0.1.*randn(rng,  Float64, (Nk,1))
# mysmoother1 = ACE.LAS(Nk ÷ 2)
mysmoother2 = ACE.LASb(Nk ÷ 5)
mysmoother3 =  ACE.LLSS(Nk ÷ 3)
mysmoother4 =  ACE.LLSSb(Nk ÷ 4)
mysmoother5 = ACE.FRSS([0.05,0.1,0.5], 0.2, 0.2)
σ = 0.4
mykernel = ACE.Kernelregression.Gaussian(σ)
reg = 1.0
mysmoother6 = ACE.NWKernelsmooth(mykernel)
mysmoothers = Array{ACE.Smoother}([mysmoother2])
# guess_parameters!(mysmoother, Nk)
myace =  ACE.ACEsim(X,Y,mysmoothers)
res = ACE.run(myace)
# Benchmark.Φ_x, Benchmark.Θ_y, Benchmark.sIx, Benchmark.sIy, Benchmark.conv_err
# h1 =  benchmark_ace_plot(Benchmark, "ACE Demo", 80) 
gr()
h1 = plot(res)
display(h1)

print_timer()
reset_timer!()