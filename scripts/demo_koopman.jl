# Copyright (c) 2021 NilsWildt
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# Load Packages
using DrWatson
@quickactivate "ACE"
include(srcdir("ACE.jl"))
using PProf
using .ACE
using Plots
include(srcdir("Smoother.jl"))
using .Smoothers
# using Suppressor
using TimerOutputs
# using ProfileView

function vec_to_matrix(x::AbstractArray{T}) where T<:Real
    return reshape(x,length(x),1)
end
const to = TimerOutput()
# ENV["JULIA_DEBUG"] = "all"
# Initalize Plot backend
gr()
# Add utils and Benchmarksuite
include(srcdir("utils.jl"))
include(srcdir("benchmark_functions.jl"))
########################################################################
########################################################################
f(x) = x^2
x = collect(range(1,10,length=500))
xdot = f.(x)
# display(plot(Benchmark.X,Benchmark.Y,dpi=80))
# s1 =ACE.Smoothers.FRSS([0.05,0.1,0.5], 0.2, 0.2)
smoother =ACE.Smoothers.LASb(length(xdot) ÷  2)
# smoother = Array{ACE.Smoothers.Smoother}([s1,smoother])
# σ = 0.03
# mykernel = Kernelregression.Gaussian(σ)
# smoother =  Array{ACE.Smoothers.Smoother}([ACE.NWKernelsmooth(mykernel)])
Simulation1 = ACE.ACEsim(Matrix(x|>vec_to_matrix),Matrix(xdot|>vec_to_matrix),smoother;multiloopversion=:fresh,itermax_inner=100,itermax_outer=100,errorbound=1e-5)
# @profile for i in 1:10
#  ACE.run(Simulation1)
# end
# result = @timeit  "acetotal" ACE.run(Simulation1)
result = @timeit to "acetotal" ACE.run(Simulation1)

# @info "results:" 1-result.r²[1] result.r²[1] 1-result.ρ result.ρ result.AARD result.r_orig result.t
# @profile  ACE.run(Simulation1)
# Profile.print()

if !isnan.(result.r²[1])
        # plotly()
  display(plot(result,dpi=80,size=(1800,900)))
else
        @error "" result.r²[1] result.ρ result.Φ_x, result.Θ_y
end

display(to)

# reset_timer!()