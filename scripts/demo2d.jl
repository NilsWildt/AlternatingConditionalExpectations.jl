# Copyright (c) 2021 NilsWildt
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# Load Packages
using DrWatson
@quickactivate "ACE"
include(srcdir("ACE.jl"))
using TrendDecomposition
using .ACE
using Plots
using PrettyTables
using FixedSizeArrays
# using Suppressor
using TimerOutputs
const to = TimerOutput()
# ENV["JULIA_DEBUG"] = "all"
# Initalize Plot backend
gr()
# Add utils and Benchmarksuite
include(srcdir("utils.jl"))
########################################################################
########################################################################
Nk = 1000 # Problem size 
Benchmark =  ACE.ACEBenchmarksMakie.f_b1(Nk, 1, "uniform", 0.1, 2.0, true, 42, true, (-5.0, 1.4))

fig = ACE.ACEBenchmarksMakie.benchmark_ace_figure(Benchmark; dark=false)
display(fig)
# display(scatter(Benchmark.X,Benchmark.Y,dpi=80))
# s1 = ACE.Smoothers.FRSS([0.05,0.1,0.5], 0.2, 0.2)
# Create smoothers with concrete types
# smoother2 = ACE.Smoothers.LLSSb(Nk ÷ 24)
# smoother1 = ACE.Smoothers.FRSS()
# smoother1 = ACE.Smoothers.LAS(Nk ÷ 12 )

# If you want to chain multiple smoothers, use a Vector with concrete element type:
# σ = 0.03
# σ = 5.0
# mykernel = ACE.Smoothers.Kernelregression.Imq(σ)
# kernel_smoother = ACE.Smoothers.NWKernelsmooth(mykernel)

# Create a concrete array type for better type inference
# Helper function to create a concrete array from mixed smoother types
function create_smoother_array(smoothers...)
    T = Union{map(typeof, smoothers)...}
    return T[smoothers...]
end

# smoothers = create_smoother_array(smoother1)

XX = Matrix(Benchmark.X)
YY = vec(Benchmark.Y)

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
    ACE.Smoothers.TrendSmoother(win; smoother_type = :ma),
    ACE.Smoothers.TrendSmoother(win; weights = weights, smoother_type = :ma_weighted),
]

results = Vector{NamedTuple}(undef, length(smoothers))

for (i, sm) in enumerate(smoothers)
    sim = ACE.ACEsim(XX, YY, sm,errorbound=1e-8,itermax_inner=5000,itermax_outer=5000)
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
best_res = ACE.ACEsim(XX, YY, best_sm) |> ACE.run
fig = ACE.plot_ace_results(best_res; full=true)
# display(fig)


# print_timer()
# reset_timer!()