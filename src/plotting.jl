# Plotting entry points.
#
# The Makie implementation lives in the package extension
# `AlternatingConditionalExpectationsMakieExt`, which loads once a Makie backend
# (CairoMakie, GLMakie, ...) is imported. These stubs give the names a home in
# the parent module so they can be exported, and produce a clear error when a
# backend has not been loaded yet.

"""
    plot_ace_results(res::ACEres; savepath=nothing, kwargs...) -> Makie.Figure

Build a diagnostic figure for an [`ACEres`](@ref): the data, the fitted
transformations and the convergence history. Returns a `Makie.Figure`; pass
`savepath` to also write it to disk (PNG/PDF/SVG).

Plotting is provided by the Makie extension — load a backend first:

```julia
using CairoMakie
plot_ace_results(res; savepath = "ace.png")
```
"""
function plot_ace_results(args...; kwargs...)
    error("plotting requires a Makie backend — run `using CairoMakie` (or `using GLMakie`) before calling plot_ace_results")
end

"""
    benchmark_ace_plot(bf::BenchmarkFunction; savepath=nothing, kwargs...) -> Makie.Figure

Scatter the benchmark data and overlay the reference transform curves stored in
`bf.plot_fcs`. Requires a Makie backend (see [`plot_ace_results`](@ref)).
"""
function benchmark_ace_plot(args...; kwargs...)
    error("plotting requires a Makie backend — run `using CairoMakie` (or `using GLMakie`) before calling benchmark_ace_plot")
end
