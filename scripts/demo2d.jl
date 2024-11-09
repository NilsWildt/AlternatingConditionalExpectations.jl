# Copyright (c) 2021 NilsWildt
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# Load Packages
using DrWatson
@quickactivate "ACE"
include(srcdir("ACE.jl"))

using .ACE
using Plots
include(srcdir("Smoother.jl"))
using .Smoothers
# using Suppressor
using TimerOutputs

# const to = TimerOutput()
# ENV["JULIA_DEBUG"] = "all"
# Initalize Plot backend
gr()
# Add utils and Benchmarksuite
include(srcdir("utils.jl"))
include(srcdir("benchmark_functions.jl"))
########################################################################
########################################################################
Nk = 500 # Problem size 
Benchmark =  f_b1(Nk, 1, "uniform", 1.0, 1.0, true, 42, true, (-5.0, 1.4))
# display(plot(Benchmark.X,Benchmark.Y,dpi=80))
s1 =Smoothers.FRSS([0.05,0.1,0.5], 0.2, 0.2)
smoother =ACE.Smoothers.LASb(Nk ÷  24)
# smoother = Array{Smoother}([smoother])
# σ = 0.03
# mykernel =Kernelregression.Gaussian(σ)
# smoother =  Array{ACE.Smoother}([ACE.NWKernelsmooth(mykernel)])
Simulation1 = ACE.ACEsim(Matrix(Benchmark.X),Matrix(Benchmark.Y),smoother)
result = @timeit  "acetotal" ACE.run(Simulation1)

# @info "results:" 1-result.r²[1] result.r²[1] 1-result.ρ result.ρ result.AARD result.r_orig result.t


if !isnan.(result.r²[1])
        plotly()
  display(plot(result,dpi=80,size=(1200,899)))
else
        @error "" result.r²[1] result.ρ result.Φ_x, result.Θ_y
end


# print_timer()
# reset_timer!()