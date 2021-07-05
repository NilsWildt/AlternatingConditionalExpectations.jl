# Copyright (c) 2021 NilsWildt
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# Load Packages
using DrWatson
@quickactivate "ACE"
using ACE
using Plots
# ENV["JULIA_DEBUG"] = "all"
# Initalize Plot backend
gr()
# Add utils and Benchmarksuite
include(srcdir("utils.jl"))
include(srcdir("benchmark_functions.jl"))
########################################################################
########################################################################
# @time begin
Nk = 250 # Problem size 
Benchmark =  f_b1(Nk, 1, "uniform", 1.0, 1.0, false, 42, false, (-2.0, 1.4))
# display(plot(Benchmark.X,Benchmark.Y,dpi=80))
@info "Dbg" Nk ÷  4
smoother = ACE.LASb(Nk ÷  4)
# smoother = Array{Smoother}([ACE.FRSS([0.05,0.1,0.5], 0.2, 0.2)])
# σ = 1.0
# mykernel = ACE.Kernelregression.Gaussian(σ)
# smoother =  Array{Smoother}([ACE.NWKernelsmooth(mykernel)])
Simulation1 =  ACE.ACEsim(Matrix(Benchmark.X),Matrix(Benchmark.Y),smoother)
result = ACE.run(Simulation1)

if !isnan.(result.r²[1])
        gr()
        display(plot(result,dpi=200))
else
        @error "Fuck" result.r²[1] result.ρ result.Φ_x, result.Θ_y
end
