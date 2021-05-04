# Copyright (c) 2021 NilsWildt
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# Load Packages
using DrWatson
@quickactivate "ACE"
using ACE
using Plots
# Initalize Plot backend
gr()
# Add utils and Benchmarksuite
include(srcdir("utils.jl"))
include(srcdir("benchmark_functions.jl"))
########################################################################
########################################################################
@time begin
Nk = 2000 # Problem size 
Benchmark =  f_b2(Nk, 1, "normal", 1.0, 0.2, true, 123, false, (-5., 5.))
smoother = Array{Smoother}([ACE.LLSSb(Nk ÷ 5)])
Simulation1 =  ACE.ACEsim(Benchmark.X,Benchmark.Y,smoother)
result = ACE.run(Simulation1)
end
gr()
display(plot(result, bottom_margin=50Plots.px, left_margin=100Plots.px))