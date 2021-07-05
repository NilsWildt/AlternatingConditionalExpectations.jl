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

function get_sortidx(X::T where T <:  AbstractArray)::Tuple{Vector{Int64},Vector{Int64}}
    N = length(X)
    # We always only look at the first dimension.
    # @infiltrate
    sort_idx =  sortperm(X) # Weill can't Sortperm on SArray
    sort_idx_back = zeros(Int64, (N,))
    sort_idx_back[sort_idx] = 1:N
    return Array{Int64,1}(sort_idx[:]), Array{Int64,1}(sort_idx_back[:])
end
 
########################################################################
########################################################################
# @time begin
Nk = 5000 # Problem size 
Benchmark =  f_b2(Nk, 1, "uniform", 1.0, 1.0, false, 42, false, (-2.0, 1.4))
X = vec(Benchmark.X)
Y = vec( Benchmark.Y)
# display(plot(Benchmark.X,Benchmark.Y,dpi=80))
@info "Dbg" Nk ÷  7
# smoother = ACE.LAS(Nk ÷  25)
# smoother = Array{Smoother}([ACE.FRSS([0.05,0.1,0.5], 0.2, 0.2)])
σ = 0.05
mykernel = ACE.Kernelregression.Gaussian(σ)
smoother =  Array{Smoother}([ACE.NWKernelsmooth(mykernel)])
# Simulation1 =  ACE.ACEsim(Matrix(Benchmark.X),Matrix(Benchmark.Y),smoother)
# result = ACE.run(Simulation1)
sIx, bsIx = get_sortidx(X)
    # SOrt response variables
sIy, bsIy = get_sortidx(vec(Y))
Ysmooth  =ACE.do_smoothing(X[sIx],Y[sIx], smoother)[bsIx]

if !any(isnan.(Ysmooth))
        gr()
        h1 = plot( sort_two_arrays_native(X,Ysmooth),dpi=200,lw=5)
        scatter!(h1,X,Y,alpha=0.05)
else
        @error "Fuck" 
end


