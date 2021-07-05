# Copyright (c) 2021 NilsWildt
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# Load Packages
using DrWatson
@quickactivate "ACE"
using ACE
using Plots
using Suppressor
# ENV["JULIA_DEBUG"] = "all"
# Initalize Plot backend
gr()
# Add utils and Benchmarksuite
include(srcdir("utils.jl"))
include(srcdir("benchmark_functions.jl"))
########################################################################
########################################################################
@time begin
Nk = 5000 # Problem size 
Benchmark =  f_b1(Nk, 1, "uniform", 1.0, 1.0, true, 42, false, (-3.0, 1.4))
# display(plot(Benchmark.X,Benchmark.Y,dpi=80))
# smoother = ACE.LASb(Nk ÷  24)
smoother = Array{Smoother}([ACE.FRSS([0.05,0.1,0.5], 0.2, 0.2)])
# σ = 0.03
# mykernel = ACE.Kernelregression.Gaussian(σ)
# smoother =  Array{Smoother}([ACE.NWKernelsmooth(mykernel)])
Simulation1 =  ACE.ACEsim(Matrix(Benchmark.X),Matrix(Benchmark.Y),smoother)
result = ACE.run(Simulation1)

end

@info "results:" 1-result.r²[1] result.r²[1] 1-result.ρ result.ρ result.AARD result.r_orig result.t


if !isnan.(result.r²[1])
        gr()
      @suppress  display(plot(result,dpi=200))
else
        @error "" result.r²[1] result.ρ result.Φ_x, result.Θ_y
end
