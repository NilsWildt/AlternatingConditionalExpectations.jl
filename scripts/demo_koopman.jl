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
using PrettyTables

function vec_to_matrix(x::AbstractArray{T}) where T<:Real
    return reshape(x,length(x),1)
end
const to = TimerOutput()
# ENV["JULIA_DEBUG"] = "all"
# Initalize Plot backend
# Add utils and Benchmarksuite
include(srcdir("utils.jl"))
# include(srcdir("benchmark_functions.jl"))
########################################################################
########################################################################
f(x) = x^2
x = collect(range(1,10,length=400)) |> vec_to_matrix
xdot = f.(x) .+ 0.5*randn(length(x))|> vec_to_matrix
# display(plot(Benchmark.X,Benchmark.Y,dpi=80))
# s1 =ACE.Smoothers.FRSS([0.05,0.1,0.5], 0.2, 0.2)
# smoother =ACE.Smoothers.LASb(length(xdot) ÷  2)
# smoother = Array{ACE.Smoothers.Smoother}([s1,smoother])
# σ = 0.03
# mykernel = Kernelregression.Gaussian(σ)
# smoother =  Array{ACE.Smoothers.Smoother}([ACE.NWKernelsmooth(mykernel)])
# Simulation1 = ACE.ACEsim(Matrix(x|>vec_to_matrix),Matrix(xdot|>vec_to_matrix),smoother;multiloopversion=:fresh,itermax_inner=100,itermax_outer=100,errorbound=1e-5)
# @profile for i in 1:10
#  ACE.run(Simulation1)
# end
# result = @timeit  "acetotal" ACE.run(Simulation1)
# result = @timeit to "acetotal" ACE.run(Simulation1)

# @info "results:" 1-result.r²[1] result.r²[1] 1-result.ρ result.ρ result.AARD result.r_orig result.t
# @profile  ACE.run(Simulation1)
# Profile.print()

# if !isnan.(result.r²[1])
#         plotly()
#   display(plot(result,dpi=80,size=(1800,900)))
# else
#         @error "" result.r²[1] result.ρ result.Φ_x, result.Θ_y
# end

# display(to)
# fig = ACE.plot_ace_results(result; full=true)
# reset_timer!()


Nk = length(x)
# Build a suite of smoothers
win = max(3, Nk ÷ 24)
weights = fill(1.0, win)
weights ./= sum(weights)

smoothers = [
    ACE.Smoothers.LAS(win),
    ACE.Smoothers.LASb(win),
    ACE.Smoothers.LLSS(win),
    ACE.Smoothers.LLSSb(win),
    ACE.Smoothers.FRSS(),
    ACE.Smoothers.NWKernelsmooth(ACE.Smoothers.Kernelregression.Imq(0.005)),
    ACE.Smoothers.NWKernelsmooth(ACE.Smoothers.Kernelregression.Imq(0.01)),
    ACE.Smoothers.NWKernelsmooth(ACE.Smoothers.Kernelregression.Gaussian(0.05)),
    ACE.Smoothers.NWKernelsmooth(ACE.Smoothers.Kernelregression.Gaussian(0.01)),
]

results = Vector{NamedTuple}(undef, length(smoothers))

for (i, sm) in enumerate(smoothers)
    sim = ACE.ACEsim(x, xdot, sm,errorbound=1e-8,itermax_inner=5000,itermax_outer=5000)
    res = @timeit to "acetotal" ACE.run(sim)
    results[i] = (
        smoother = String(sm),
        r2 = res.r²[1],
        rho = res.ρ,
        one_minus_r2 = 1 - res.r²[1],
        one_minus_rho = 1 - res.ρ,
        AARD = res.AARD,
        time_s = res.t,
        iters = res.itercount,
    )
#     display(res)
end

# Pretty table output
header = [
    "Smoother", "r²", "1-r²", "ρ", "1-ρ", "AARD", "time [s]", "iters"
]
table = Array{Any}(undef, length(results), length(header))
for (i, r) in enumerate(results)
    table[i, 1] = r.smoother
    table[i, 2] = round(r.r2; digits=6)
    table[i, 3] = round(r.one_minus_r2; digits=6)
    table[i, 4] = round(r.rho; digits=6)
    table[i, 5] = round(r.one_minus_rho; digits=6)
    table[i, 6] = round(r.AARD; digits=6)
    table[i, 7] = round(r.time_s; digits=4)
    table[i, 8] = r.iters
end

pretty_table(table; header = header, alignment = :l, tf = tf_compact)

# Optionally show the best result's plots
best_idx = argmin([r.one_minus_r2 for r in results])
best_sm = smoothers[best_idx]
best_res = ACE.ACEsim(x, xdot, best_sm) |> ACE.run
fig = ACE.plot_ace_results(best_res; full=true)
# display(fig)