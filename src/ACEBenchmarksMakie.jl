module ACEBenchmarksMakie

using Random
using DocStringExtensions
using LaTeXStrings
using Colors
using CairoMakie 
# or GLMakie/WGLMakie — CairoMakie is great for docs/PNG/PDF
# using Documenter   # enable in your docs build, see "Docs setup" below
# using Latexify     # still supported if you want latexify labels from plot_fcs
# using PrettyTables
# using Parameters
# include(srcdir("utils.jl")) # keep if you rely on it elsewhere

# -------------------------
# Core types & utilities
# -------------------------

"""
Abstract supertype for all ACE benchmark problems.
"""
abstract type BenchmarkFunction end

"""
    heaviside(x)

Heaviside step function evaluated elementwise.

$(TYPEDSIGNATURES)
"""
heaviside(x) = @. 0.5 * (sign(x) + 1.0)

"""
    normal_sample(numSamples, numDim, μ, σ; use_seed=false, seed=0)

Gaussian sampler (matrix of size `(numSamples, numDim)`).

$(TYPEDSIGNATURES)
"""
function normal_sample(numSamples::Int, numDim::Int, μ::Float64, σ::Float64,
                       use_seed::Bool=false, seed::Int=0)
    dims = (numSamples, numDim)
    rng = use_seed ? MersenneTwister(seed) : MersenneTwister()
    return σ .* randn(rng, Float64, dims) .+ μ
end

"""
    uniform_sample(numSamples, numDim, σ_noise, use_seed, seed=0; bounds=(0.0, 1.0))

Uniform sampler on `bounds` (matrix of size `(numSamples, numDim)`).

`σ_noise` is accepted for signature compatibility but not used here.

$(TYPEDSIGNATURES)
"""
function uniform_sample(numSamples::Int, numDim::Int, σ_noise::Float64, use_seed::Bool,
                        seed::Int=0, bounds::Tuple{Float64,Float64}=(0.0, 1.0))
    lb, ub = bounds
    dims = (numSamples, numDim)
    rng = use_seed ? MersenneTwister(seed) : MersenneTwister()
    return (ub - lb) .* rand(rng, Float64, dims) .+ lb
end

"""
    get_sample(numSamples, numDim, samplingmethod, σ; use_seed=false, seed=0, bounds=(0,1))

Dispatch to `uniform_sample` or `normal_sample` depending on `samplingmethod ∈ {"uniform","normal"}`.

$(TYPEDSIGNATURES)
"""
function get_sample(numSamples::Int, numDim::Int, samplingmethod::String, σ::Float64,
                    use_seed::Bool=false, seed::Int=0,
                    bounds::Tuple{Float64,Float64}=(0.0, 1.0))
    if samplingmethod == "uniform"
        return uniform_sample(numSamples, numDim, σ, use_seed, seed, bounds)
    elseif samplingmethod == "normal"
        return normal_sample(numSamples, numDim, 0.0, σ, use_seed, seed)
    else
        @warn "Unknown sampling method '$samplingmethod'. Falling back to 'normal'."
        return normal_sample(numSamples, numDim, 0.0, 1.0, use_seed, seed)
    end
end

normalize(x::Float64, a::Float64, b::Float64) = (x - a) / (b - a)

# -------------------------
# Benchmark definitions
# -------------------------

# Common fields are kept; Makie plotting will read X, Y, Φ_x, Θ_y, plot_view_bounds, plot_fcs, etc.

mutable struct f_b1 <: BenchmarkFunction
    numSamples::Int
    numDim::Int
    samplingmethod::String
    σ_x::Float64
    σ_noise::Float64
    use_seed::Bool
    seed::Int
    scale_data::Bool
    bounds::Tuple{Float64,Float64}
    X::AbstractArray
    Y::AbstractArray
    plot_view_bounds::AbstractArray
    plot_fcs::AbstractArray
    name::String
    scale_factors::AbstractArray
    Φ_x::AbstractArray
    Θ_y::AbstractArray
    sIx::AbstractArray
    sIy::AbstractArray
    bsIx::AbstractArray
    bsIy::AbstractArray
    conv_err::AbstractArray

    function f_b1(numSamples::Int, numDim::Int, samplingmethod::String,
                  σ_x::Float64, σ_noise::Float64, use_seed::Bool, seed::Int,
                  scale_data::Bool, bounds::Tuple{Float64,Float64}=(0.0, 1.0))
        lb, ub = bounds
        X = get_sample(numSamples, numDim, samplingmethod, σ_x, use_seed, seed, bounds)
        Y = exp.(X .^ 3 .+ normal_sample(numSamples, numDim, 0.0, σ_noise, use_seed, seed + 1))
        viewbounds = [[(-2.5, 2.5), (0, 80)], [(-2.5, 2.5), (-2.5, 2.5)],
                      [(0, 80), (-2, 5)], [(-3, 3), (-3, 3)]]
        scale_factors = [zeros(2), zeros(2)]
        if scale_data
            xmax, xmin = maximum(X), minimum(X)
            ymax, ymin = maximum(Y), minimum(Y)
            scale_factors = [[xmin, xmax], [ymin, ymax]]
            X = normalize.(X, xmin, xmax)
            Y = normalize.(Y, ymin, ymax)
            viewbounds = [[(0, 1), (0, 1)], [(0, 1), (0, 1)], [(0, 1.0), (0, 1.5)], [(-3, 3), (-3, 3)]]
        end
        plot_fcs = [Dict(L"x^3" => x -> x .^ 3, L"x" => x -> x),
                    Dict(L"\log(x)" => x -> log.(x), L"\sqrt[3]{\log(x)}" => x -> (log.(x)).^(1/3))]
        new(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data,
            bounds, X, Y, viewbounds, plot_fcs,
            L"BenchmarkFunction 1: $f_{b1}(x_1)= x_2 = \exp{(x_1^3 + \varepsilon)}$ ",
            scale_factors, Float64[], Float64[], Int[], Int[], Int[], Int[], Float64[])
    end
end

mutable struct f_b2 <: BenchmarkFunction
    numSamples::Int
    numDim::Int
    samplingmethod::String
    σ_x::Float64
    σ_noise::Float64
    use_seed::Bool
    seed::Int
    scale_data::Bool
    bounds::Tuple{Float64,Float64}
    X::AbstractArray
    Y::AbstractArray
    plot_view_bounds::AbstractArray
    plot_fcs::AbstractArray
    name::String
    scale_factors::AbstractArray
    Φ_x::AbstractArray
    Θ_y::AbstractArray
    sIx::AbstractArray
    sIy::AbstractArray
    bsIx::AbstractArray
    bsIy::AbstractArray
    conv_err::AbstractArray

    function f_b2(numSamples::Int, numDim::Int, samplingmethod::String,
                  σ_x::Float64, σ_noise::Float64, use_seed::Bool, seed::Int,
                  scale_data::Bool, bounds::Tuple{Float64,Float64}=(0.0, 1.0))
        lb, ub = bounds
        X = get_sample(numSamples, numDim, samplingmethod, σ_x, use_seed, seed, bounds)
        Y = exp.(sin.(X) .+ normal_sample(numSamples, numDim, 0.0, σ_noise, use_seed, seed + 1))
        viewbounds = [[(lb, ub), (0, 40)], [(lb, ub), (-ub, ub)], [(0, 80), (-2, 5)], [(-3, 3), (-3, 3)]]
        scale_factors = [zeros(2), zeros(2)]
        if scale_data
            xmax, xmin = maximum(X), minimum(X)
            ymax, ymin = maximum(Y), minimum(Y)
            scale_factors = [[xmin, xmax], [ymin, ymax]]
            X = normalize.(X, xmin, xmax)
            Y = normalize.(Y, ymin, ymax)
            viewbounds = [[(0, 1), (0, 1)], [(-2.0, 2.0), (-2.0, 2.0)], [(-2.0, 2.0), (-2.0, 2.0)], [(-3, 3), (-3, 3)]]
        end
        plot_fcs = [Dict("sin(x)" => x -> sin.(x)), Dict("log(x)" => x -> log.(x))]
        new(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data,
            bounds, X, Y, viewbounds, plot_fcs,
            L"BenchmarkFunction 2: $f_{b2}(x_1)= x_2 = \exp{(\sin{(x_1)} + \varepsilon)}$ ",
            scale_factors, Float64[], Float64[], Int[], Int[], Int[], Int[], Float64[])
    end
end

mutable struct f_b3 <: BenchmarkFunction
    numSamples::Int
    numDim::Int
    samplingmethod::String
    σ_x::Float64
    σ_noise::Float64
    use_seed::Bool
    seed::Int
    scale_data::Bool
    bounds::Tuple{Float64,Float64}
    X::AbstractArray
    Y::AbstractArray
    plot_view_bounds::AbstractArray
    plot_fcs::AbstractArray
    name::String
    scale_factors::AbstractArray
    Φ_x::AbstractArray
    Θ_y::AbstractArray
    sIx::AbstractArray
    sIy::AbstractArray
    bsIx::AbstractArray
    bsIy::AbstractArray
    conv_err::AbstractArray

    function f_b3(numSamples::Int, numDim::Int, samplingmethod::String,
                  σ_x::Float64, σ_noise::Float64, use_seed::Bool, seed::Int,
                  scale_data::Bool, bounds::Tuple{Float64,Float64}=(0.0, 1.0))
        lb, ub = bounds
        X = get_sample(numSamples, numDim, samplingmethod, σ_x, use_seed, seed, bounds)
        Y = 2 .* heaviside.(X) .+ normal_sample(numSamples, numDim, 0.0, σ_noise, use_seed, seed + 1)
        viewbounds = [[(lb, ub), (-1.3, 5)], [(lb, ub), (lb, ub)], [(-5, 5), (-5, 5)], [(-3, 3), (-3, 3)]]
        scale_factors = [zeros(2), zeros(2)]
        if scale_data
            xmax, xmin = maximum(X), minimum(X)
            ymax, ymin = maximum(Y), minimum(Y)
            scale_factors = [[xmin, xmax], [ymin, ymax]]
            X = normalize.(X, xmin, xmax)
            Y = normalize.(Y, ymin, ymax)
            viewbounds = [[(0, 1), (0, 1)], [(0, 1), (0, 1)], [(0, 1), (0, 1)], [(-3, 3), (-3, 3)]]
        end
        plot_fcs = [] # could be Dict("σ(x)" => x -> heaviside.(x))
        new(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data,
            bounds, X, Y, viewbounds, plot_fcs,
            L"BenchmarkFunction 3: $f_{b3}(x_1)= x_2 = \sigma(x_1)$ ",
            scale_factors, Float64[], Float64[], Int[], Int[], Int[], Int[], Float64[])
    end
end

mutable struct f_b4 <: BenchmarkFunction
    numSamples::Int
    numDim::Int
    samplingmethod::String
    σ_x::Float64
    σ_noise::Float64
    use_seed::Bool
    seed::Int
    scale_data::Bool
    bounds::Tuple{Float64,Float64}
    X::AbstractArray
    Y::AbstractArray
    plot_view_bounds::AbstractArray
    plot_fcs::AbstractArray
    name::String
    scale_factors::AbstractArray
    Φ_x::AbstractArray
    Θ_y::AbstractArray
    sIx::AbstractArray
    sIy::AbstractArray
    bsIx::AbstractArray
    bsIy::AbstractArray
    conv_err::AbstractArray

    function f_b4(numSamples::Int, numDim::Int, samplingmethod::String,
                  σ_x::Float64, σ_noise::Float64, use_seed::Bool, seed::Int,
                  scale_data::Bool, bounds::Tuple{Float64,Float64}=(0.0, 1.0))
        lb, ub = bounds
        X = get_sample(numSamples, numDim, samplingmethod, σ_x, use_seed, seed, bounds)
        Y = log.(abs.(X .+ normal_sample(numSamples, numDim, 0.0, σ_noise, use_seed, seed + 1)))
        viewbounds = [[(lb, ub), (-5, 5)],
                      [(lb, ub), (-5, ub)],
                      [(lb, ub), (-2, 5)],
                      [(-2.5, 2.5), (-2.5, 2.5)]]
        scale_factors = [zeros(2), zeros(2)]
        if scale_data
            xmax, xmin = maximum(X), minimum(X)
            ymax, ymin = maximum(Y), minimum(Y)
            scale_factors = [[xmin, xmax], [ymin, ymax]]
            X = normalize.(X, xmin, xmax)
            Y = normalize.(Y, ymin, ymax)
            viewbounds = [[(0, 1), (0, 1)], [(0, 1), (0, 1)], [(0, 1), (0, 1)], [(-3, 3), (-3, 3)]]
        end
        plot_fcs = [Dict("abs(x)" => x -> abs.(x)), Dict("exp(x)" => x -> exp.(x))]
        new(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data,
            bounds, X, Y, viewbounds, plot_fcs,
            L"BenchmarkFunction 4: $f_{b4}(x_1)= x_2 = \log(\,|x_1 + \\varepsilon|\,)$ ",
            scale_factors, Float64[], Float64[], Int[], Int[], Int[], Int[], Float64[])
    end
end

mutable struct f_toy <: BenchmarkFunction
    numSamples::Int
    numDim::Int
    samplingmethod::String
    σ_x::Float64
    σ_noise::Float64
    use_seed::Bool
    seed::Int
    scale_data::Bool
    bounds::Tuple{Float64,Float64}
    X::AbstractArray
    Y::AbstractArray
    plot_view_bounds::AbstractArray
    plot_fcs::AbstractArray
    name::String
    scale_factors::AbstractArray
    Φ_x::AbstractArray
    Θ_y::AbstractArray
    sIx::AbstractArray
    sIy::AbstractArray
    bsIx::AbstractArray
    bsIy::AbstractArray
    conv_err::AbstractArray

    function f_toy(numSamples::Int, numDim::Int, samplingmethod::String,
                   σ_x::Float64, σ_noise::Float64, use_seed::Bool, seed::Int,
                   scale_data::Bool, bounds::Tuple{Float64,Float64}=(0.0, 1.0))
        lb, ub = bounds
        X = get_sample(numSamples, numDim, samplingmethod, σ_x, use_seed, seed, bounds)
        Y = X .+ normal_sample(numSamples, numDim, 0.0, σ_noise, use_seed, seed + 1)
        viewbounds = [[(lb, ub), (-5, 5)],
                      [(lb, ub), (-5, ub)],
                      [(lb, ub), (-2, 5)],
                      [(-2.5, 2.5), (-2.5, 2.5)]]
        scale_factors = [zeros(2), zeros(2)]
        if scale_data
            xmax, xmin = maximum(X), minimum(X)
            ymax, ymin = maximum(Y), minimum(Y)
            scale_factors = [[xmin, xmax], [ymin, ymax]]
            X = normalize.(X, xmin, xmax)
            Y = normalize.(Y, ymin, ymax)
            viewbounds = [[(0, 1), (0, 1)], [(0, 1), (0, 1)], [(0, 1), (0, 1)], [(-3, 3), (-3, 3)]]
        end
        plot_fcs = [Dict("x" => x -> x), Dict("x" => x -> x)]
        new(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data,
            bounds, X, Y, viewbounds, plot_fcs,
            L"Toy: $f_{toy} = X + \\text{noise}$",
            scale_factors, Float64[], Float64[], Int[], Int[], Int[], Int[], Float64[])
    end
end

# Keep show methods minimal; Documenter will render fields nicely.
Base.show(io::IO, bf::BenchmarkFunction) = print(io, bf.name)
function Base.show(io::IO, ::MIME"text/html", bf::BenchmarkFunction)
    print(io, bf.name)
end

# -------------------------
# Makie plotting
# -------------------------

"""
    ace_figure(bf; title=nothing, figure_resolution=(1120, 840), dark=true, scatter_size=2)

Create a Makie `Figure` with the 2×2 ACE diagnostic panels:

1. `X` vs `Y`
2. `X` vs `Φ_x`
3. `Y` vs `Θ_y`
4. `Φ_x` vs `Θ_y`

Also overlays any function curves found in `bf.plot_fcs` (sorted inputs) on panels (2) and (3).
If `bf.conv_err` is present (non-empty), `benchmark_ace_figure` (below) adds a 5th panel.

If a function `ε²(Φ_x_subset, Θ_y_subset)` is defined in scope and `bf.sIx`, `bf.sIy` are set,
its value is appended to the figure title.

$(TYPEDSIGNATURES)
"""
function ace_figure(bf::BenchmarkFunction; title=nothing,
                    figure_resolution=(1120, 840), dark::Bool=true, scatter_size=2)
    # Theme
    fig_bg = dark ? Makie.RGBf(0.2, 0.2, 0.2) : Makie.RGBf(1, 1, 1)
    axis_fg = dark ? :white : :black
    axis_bg = dark ? Makie.RGBf(0.15, 0.15, 0.15) : Makie.RGBf(1, 1, 1)

    fig = Figure(resolution=figure_resolution, backgroundcolor=fig_bg, figure_padding=10)

    # Title (optional)
    if title === nothing
        # Try to compute ε² if user has defined it and sIx/sIy exist:
        e2txt = nothing
        if isdefined(Main, :ε²) && !isempty(getfield(bf, :Φ_x)) && !isempty(getfield(bf, :Θ_y)) &&
           !isempty(getfield(bf, :sIx)) && !isempty(getfield(bf, :sIy))
            try
                ε² = getfield(Main, :ε²)
                e2 = round(abs(ε²(getfield(bf, :Φ_x)[getfield(bf, :sIx)],
                                  getfield(bf, :Θ_y)[getfield(bf, :sIy)])); digits=4)
                e2txt = "ε² = $(e2)"
            catch
                e2txt = nothing
            end
        end
        tstr = e2txt === nothing ? "ACE result" : "ACE result — $(e2txt)"
        Label(fig[1, 1:2], tstr; tellwidth=false, color=axis_fg, fontsize=18)
    else
        Label(fig[1, 1:2], string(title); tellwidth=false, color=axis_fg, fontsize=18)
    end

    # Grid 2×2
    gl = fig[2, 1:2] = GridLayout()

    X = bf.X; Y = bf.Y
    Φx = getfield(bf, :Φ_x)
    Θy = getfield(bf, :Θ_y)
    vb = bf.plot_view_bounds

    # Panel 1: X vs Y
    ax1 = Axis(gl[1, 1], xlabel="X", ylabel="Y", backgroundcolor=axis_bg)
    if !isempty(vb) && length(vb) ≥ 1
        xlims!(ax1, vb[1][1]...); ylims!(ax1, vb[1][2]...)
    end
    scatter!(ax1, vec(X), vec(Y); markersize=scatter_size)

    # Panel 2: X vs Φ_x
    ax2 = Axis(gl[1, 2], xlabel="X", ylabel=L"\Phi(X)", backgroundcolor=axis_bg)
    if !isempty(vb) && length(vb) ≥ 2
        xlims!(ax2, vb[2][1]...); ylims!(ax2, vb[2][2]...)
    end
    if !isempty(Φx)
        scatter!(ax2, vec(X), vec(Φx); markersize=scatter_size)
        # overlay curves if provided
        if length(bf.plot_fcs) ≥ 1 && !isempty(bf.plot_fcs[1])
            xs = sort(vec(X))
            for (lab, f) in bf.plot_fcs[1]
                lines!(ax2, xs, f(xs), label=String(lab))
            end
            axislegend(ax2; position=:rb, framevisible=false, labelcolor=axis_fg)
        end
    end

    # Panel 3: Y vs Θ_y
    ax3 = Axis(gl[2, 1], xlabel="Y", ylabel=L"\Theta(Y)", backgroundcolor=axis_bg)
    if !isempty(vb) && length(vb) ≥ 3
        xlims!(ax3, vb[3][1]...); ylims!(ax3, vb[3][2]...)
    end
    if !isempty(Θy)
        scatter!(ax3, vec(Y), vec(Θy); markersize=scatter_size)
        if length(bf.plot_fcs) ≥ 2 && !isempty(bf.plot_fcs[2])
            ys = sort(vec(Y))
            for (lab, f) in bf.plot_fcs[2]
                # mirror the original behavior where some curves used transforms:
                vals = f(ys)
                try
                    lines!(ax3, ys, vals, label=String(lab))
                catch
                    # fall back to broadcasting if f expects elementwise
                    lines!(ax3, ys, f.(ys), label=String(lab))
                end
            end
            axislegend(ax3; position=:rb, framevisible=false, labelcolor=axis_fg)
        end
    end

    # Panel 4: Φ_x vs Θ_y
    ax4 = Axis(gl[2, 2], xlabel=L"\Phi(X)", ylabel=L"\Theta(Y)", backgroundcolor=axis_bg)
    if !isempty(vb) && length(vb) ≥ 4
        xlims!(ax4, vb[4][1]...); ylims!(ax4, vb[4][2]...)
    end
    if !isempty(Φx) && !isempty(Θy)
        scatter!(ax4, vec(Φx), vec(Θy); markersize=scatter_size)
    end

    fig
end

"""
    benchmark_ace_figure(bf; title=nothing, figure_resolution=(1120, 980), dark=true, scatter_size=2)

Makie variant of your `benchmark_ace_plot`. Builds the 2×2 diagnostic grid and,
if `bf.conv_err` is non-empty, appends a 5th panel with a simple convergence trace.

Returns the Makie `Figure`.

$(TYPEDSIGNATURES)
"""
function benchmark_ace_figure(bf::BenchmarkFunction; title=nothing,
                              figure_resolution=(1120, 980), dark::Bool=true, scatter_size=2)
    # Build the 2×2 portion first
    # Use a taller figure to fit the convergence panel
    fig = ace_figure(bf; title=title, figure_resolution=(figure_resolution[1], figure_resolution[2]-140),
                     dark=dark, scatter_size=scatter_size)

    # Add convergence plot (row below the grid) if available
    conv = getfield(bf, :conv_err)
    if !isempty(conv)
        fig[3, 1:2] = GridLayout()
        fig_bg = fig.backgroundcolor[]
        axis_fg = (fig_bg.r + fig_bg.g + fig_bg.b) < 1.5 ? :white : :black
        axis_bg = fig_bg
        ax5 = Axis(fig[3, 1:2], xlabel="Iteration", ylabel="Error",
                   backgroundcolor=axis_bg)
        lines!(ax5, 1:length(conv), conv)
    end

    fig
end

# -------------------------
# Saving helpers (optional)
# -------------------------

"""
    save_ace_figure(path, bf; kwargs...)

Convenience wrapper around `save(path, ace_figure(bf; kwargs...))`.

$(TYPEDSIGNATURES)
"""
function save_ace_figure(path::AbstractString, bf::BenchmarkFunction; kwargs...)
    fig = ace_figure(bf; kwargs...)
    save(path, fig)
    return path
end

"""
    save_benchmark_ace_figure(path, bf; kwargs...)

Convenience wrapper around `save(path, benchmark_ace_figure(bf; kwargs...))`.

$(TYPEDSIGNATURES)
"""
function save_benchmark_ace_figure(path::AbstractString, bf::BenchmarkFunction; kwargs...)
    fig = benchmark_ace_figure(bf; kwargs...)
    save(path, fig)
    return path
end

# -------------------------
# Latex string renderer (optional utility)
# -------------------------

"""
    render(s::LaTeXString; debug=false, name=tempname(), command="\\Large")

Render a LaTeXString to a standalone PDF and open it (from Latexify.jl docs utility pattern).

$(TYPEDSIGNATURES)
"""
function render(s::LaTeXString; debug=false, name=tempname(), command="\\Large")
    doc = """
    \\documentclass[varwidth=100cm]{standalone}
    \\usepackage{amssymb}
    \\usepackage{amsmath}
    $(occursin("\\ce{", s) ? "\\usepackage{mhchem}" : "")
    \\begin{document}
    {
        $command
        $s
    }
    \\end{document}
    """
    doc = replace(doc, "\\begin{align}" => "\\[\\n\\begin{aligned}")
    doc = replace(doc, "\\end{align}"   => "\\end{aligned}\\n\\]")
    doc = replace(doc, "\\require{mhchem}\\n" => "")
    open("$(name).tex", "w") do f
        write(f, doc)
    end
    cd(dirname(name)) do
        cmd = `lualatex --interaction=batchmode $(name).tex`
        debug || (cmd = pipeline(cmd, devnull))
        run(cmd)
    end
    if Sys.iswindows()
        run(`cmd /c "start $(name).pdf"`, wait=false)
    elseif Sys.islinux() || Sys.isbsd()
        run(`xdg-open $(name).pdf`, wait=false)
    elseif Sys.isapple()
        run(`open $(name).pdf`, wait=false)
    end
    return nothing
end

# -------------------------
# Exports
# -------------------------
export BenchmarkFunction, f_b1, f_b2, f_b3, f_b4, f_toy
export heaviside, normal_sample, uniform_sample, get_sample
export ace_figure, benchmark_ace_figure, save_ace_figure, save_benchmark_ace_figure

end # module
