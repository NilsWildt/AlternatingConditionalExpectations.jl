# Benchmark functions for ACE (ported and cleaned from the original ACE.jl).

"""
    BenchmarkFunction

Abstract supertype for ACE benchmark problems.
"""
abstract type BenchmarkFunction end

"""
    normalize(x, a, b)

Linear normalization of `x` into `[0, 1]` using range `[a, b]`.
"""
normalize(x::Real, a::Real, b::Real) = (x - a) / (b - a)

"""
    normal_sample(numSamples, numDim, μ, σ, use_seed, seed)

Draw `numSamples × numDim` samples from `N(μ, σ²)`.
"""
function normal_sample(numSamples::Int, numDim::Int, μ::Float64, σ::Float64,
                       use_seed::Bool = false, seed::Int = 0)
    dims = (numSamples, numDim)
    rng = use_seed ? MersenneTwister(seed) : MersenneTwister()
    return σ .* randn(rng, Float64, dims) .+ μ
end

"""
    uniform_sample(numSamples, numDim, σ_noise, use_seed, seed, bounds)

Draw `numSamples × numDim` samples uniformly in `bounds`.
"""
function uniform_sample(numSamples::Int, numDim::Int, σ_noise::Float64, use_seed::Bool,
                        seed::Int = 0, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
    lb, ub = bounds
    dims = (numSamples, numDim)
    rng = use_seed ? MersenneTwister(seed) : MersenneTwister()
    return abs(ub - lb) .* rand(rng, Float64, dims) .+ lb
end

"""
    get_sample(numSamples, numDim, samplingmethod, σ, use_seed, seed, bounds)

Draw samples using either `"uniform"` or `"normal"` sampling.
"""
function get_sample(numSamples::Int, numDim::Int, samplingmethod::String, σ::Float64,
                    use_seed::Bool, seed::Int = 0, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
    if samplingmethod == "uniform"
        return uniform_sample(numSamples, numDim, σ, use_seed, seed, bounds)
    elseif samplingmethod == "normal"
        return normal_sample(numSamples, numDim, 0.0, σ, use_seed, seed)
    else
        @warn "Unknown sampling method '$samplingmethod'. Using normal distributed samples."
        return normal_sample(numSamples, numDim, 0.0, 1.0, use_seed, seed)
    end
end

# Shared fields for all benchmarks: the result arrays (`Φ_x`, `Θ_y`, indices,
# convergence) are filled in by `ace_run` or left empty.

"""
    f_b1(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data, bounds)

Benchmark 1: `Y = exp(X³ + ε)`.
"""
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

    function f_b1(numSamples::Int, numDim::Int, samplingmethod::String, σ_x::Float64,
                  σ_noise::Float64, use_seed::Bool, seed::Int, scale_data::Bool,
                  bounds::Tuple{Float64,Float64} = (0.0, 1.0))
        lb, ub = bounds
        X = get_sample(numSamples, numDim, samplingmethod, σ_x, use_seed, seed, bounds)
        Y = exp.(X .^ 3 .+ normal_sample(numSamples, numDim, 0.0, σ_noise, use_seed, seed + 1))
        viewbounds = [[(-2.5, 2.5), (0, 80)], [(-2.5, 2.5), (-2.5, 2.5)], [(0, 80), (-2, 5)], [(-3, 3), (-3, 3)]]
        scale_factors = [zeros(2), zeros(2)]
        if scale_data
            xmin, xmax = minimum(X), maximum(X)
            ymin, ymax = minimum(Y), maximum(Y)
            scale_factors = [[xmin, xmax], [ymin, ymax]]
            X = normalize.(X, xmin, xmax)
            Y = normalize.(Y, ymin, ymax)
            viewbounds = [[(0, 1), (0, 1)], [(0, 1), (0, 1)], [(0, 1.0), (0, 1.5)], [(-3, 3), (-3, 3)]]
        end
        plot_fcs = [Dict(L"x^3" => x -> x .^ 3, L"x" => x -> x),
                    Dict(L"\log(x)" => x -> log.(x), L"3throot\log(x)" => x -> log.(x .^ (1 / 3)))]
        new(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data, bounds,
            X, Y, viewbounds, plot_fcs,
            L"BenchmarkFunction 1: $f_{b1}(x_1)= x_2 = \exp{(x_1^3 + \varepsilon)}$", scale_factors,
            Float64[], Float64[], Int64[], Int64[], Int64[], Int64[], Float64[])
    end
end

"""
    f_b2(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data, bounds)

Benchmark 2: `Y = exp(sin(X) + ε)`.
"""
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

    function f_b2(numSamples::Int, numDim::Int, samplingmethod::String, σ_x::Float64,
                  σ_noise::Float64, use_seed::Bool, seed::Int, scale_data::Bool,
                  bounds::Tuple{Float64,Float64} = (0.0, 1.0))
        lb, ub = bounds
        X = get_sample(numSamples, numDim, samplingmethod, σ_x, use_seed, seed, bounds)
        Y = exp.(sin.(X) .+ normal_sample(numSamples, numDim, 0.0, σ_noise, use_seed, seed + 1))
        viewbounds = [[(lb, ub), (0, 40)], [(lb, ub), (-ub, ub)], [(0, 80), (-2, 5)], [(-3, 3), (-3, 3)]]
        scale_factors = [zeros(2), zeros(2)]
        if scale_data
            xmin, xmax = minimum(X), maximum(X)
            ymin, ymax = minimum(Y), maximum(Y)
            scale_factors = [[xmin, xmax], [ymin, ymax]]
            X = normalize.(X, xmin, xmax)
            Y = normalize.(Y, ymin, ymax)
            viewbounds = [[(0, 1), (0, 1)], [(-2.0, 2.0), (-2.0, 2.0)], [(-2.0, 2.0), (-2.0, 2.0)], [(-3, 3), (-3, 3)]]
        end
        plot_fcs = [Dict("sin(x)" => x -> sin.(x)), Dict("log(x)" => x -> log.(x))]
        new(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data, bounds,
            X, Y, viewbounds, plot_fcs,
            L"BenchmarkFunction 2: $f_{b2}(x_1)= x_2 = \exp{(\sin{(x_1)} + \varepsilon)}$", scale_factors,
            Float64[], Float64[], Int64[], Int64[], Int64[], Int64[], Float64[])
    end
end

"""
    f_b3(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data, bounds)

Benchmark 3: `Y = 2·H(X) + ε` (Heaviside step).
"""
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

    function f_b3(numSamples::Int, numDim::Int, samplingmethod::String, σ_x::Float64,
                  σ_noise::Float64, use_seed::Bool, seed::Int, scale_data::Bool,
                  bounds::Tuple{Float64,Float64} = (0.0, 1.0))
        lb, ub = bounds
        X = get_sample(numSamples, numDim, samplingmethod, σ_x, use_seed, seed, bounds)
        Y = 2.0 .* heaviside.(X) .+ normal_sample(numSamples, numDim, 0.0, σ_noise, use_seed, seed + 1)
        viewbounds = [[(lb, ub), (-1.3, 5)], [(lb, ub), (lb, ub)], [(-5, 5), (-5, 5)], [(-3, 3), (-3, 3)]]
        scale_factors = [zeros(2), zeros(2)]
        if scale_data
            xmin, xmax = minimum(X), maximum(X)
            ymin, ymax = minimum(Y), maximum(Y)
            scale_factors = [[xmin, xmax], [ymin, ymax]]
            X = normalize.(X, xmin, xmax)
            Y = normalize.(Y, ymin, ymax)
            viewbounds = [[(0, 1), (0, 1)], [(0, 1), (0, 1)], [(0, 1), (0, 1)], [(-3, 3), (-3, 3)]]
        end
        plot_fcs = []
        new(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data, bounds,
            X, Y, viewbounds, plot_fcs,
            L"BenchmarkFunction 3: $f_{b3}(x_1)= x_2 = \sigma(x_2)$", scale_factors,
            Float64[], Float64[], Int64[], Int64[], Int64[], Int64[], Float64[])
    end
end

"""
    f_b4(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data, bounds)

Benchmark 4: `Y = log(|X + ε|)`.
"""
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

    function f_b4(numSamples::Int, numDim::Int, samplingmethod::String, σ_x::Float64,
                  σ_noise::Float64, use_seed::Bool, seed::Int, scale_data::Bool,
                  bounds::Tuple{Float64,Float64} = (0.0, 1.0))
        lb, ub = bounds
        X = get_sample(numSamples, numDim, samplingmethod, σ_x, use_seed, seed, bounds)
        Y = log.(abs.(X .+ normal_sample(numSamples, numDim, 0.0, σ_noise, use_seed, seed + 1)))
        viewbounds = [[(lb, ub), (-5, 5)], [(lb, ub), (-5, ub)], [(lb, ub), (-2, 5)], [(-2.5, 2.5), (-2.5, 2.5)]]
        scale_factors = [zeros(2), zeros(2)]
        if scale_data
            xmin, xmax = minimum(X), maximum(X)
            ymin, ymax = minimum(Y), maximum(Y)
            scale_factors = [[xmin, xmax], [ymin, ymax]]
            X = normalize.(X, xmin, xmax)
            Y = normalize.(Y, ymin, ymax)
            viewbounds = [[(0, 1), (0, 1)], [(0, 1), (0, 1)], [(0, 1), (0, 1)], [(-3, 3), (-3, 3)]]
        end
        plot_fcs = [Dict("abs(x)" => x -> abs.(x)), Dict("exp(x)" => x -> exp.(x))]
        new(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data, bounds,
            X, Y, viewbounds, plot_fcs,
            L"BenchmarkFunction 4: $f_{b4}(x_1)= x_2 = \log{(\log{(x_2)})}$", scale_factors,
            Float64[], Float64[], Int64[], Int64[], Int64[], Int64[], Float64[])
    end
end

"""
    f_toy(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data, bounds)

Toy benchmark: `Y = X + ε`.
"""
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

    function f_toy(numSamples::Int, numDim::Int, samplingmethod::String, σ_x::Float64,
                   σ_noise::Float64, use_seed::Bool, seed::Int, scale_data::Bool,
                   bounds::Tuple{Float64,Float64} = (0.0, 1.0))
        lb, ub = bounds
        X = get_sample(numSamples, numDim, samplingmethod, σ_x, use_seed, seed, bounds)
        Y = X .+ normal_sample(numSamples, numDim, 0.0, σ_noise, use_seed, seed + 1)
        viewbounds = [[(lb, ub), (-5, 5)], [(lb, ub), (-5, ub)], [(lb, ub), (-2, 5)], [(-2.5, 2.5), (-2.5, 2.5)]]
        scale_factors = [zeros(2), zeros(2)]
        if scale_data
            xmin, xmax = minimum(X), maximum(X)
            ymin, ymax = minimum(Y), maximum(Y)
            scale_factors = [[xmin, xmax], [ymin, ymax]]
            X = normalize.(X, xmin, xmax)
            Y = normalize.(Y, ymin, ymax)
            viewbounds = [[(0, 1), (0, 1)], [(0, 1), (0, 1)], [(0, 1), (0, 1)], [(-2.5, 2.5), (-2.5, 2.5)]]
        end
        plot_fcs = [Dict("x" => x -> x), Dict("x" => x -> x)]
        new(numSamples, numDim, samplingmethod, σ_x, σ_noise, use_seed, seed, scale_data, bounds,
            X, Y, viewbounds, plot_fcs,
            L"Toy function: $f_{toy} = X + \varepsilon$", scale_factors,
            Float64[], Float64[], Int64[], Int64[], Int64[], Int64[], Float64[])
    end
end

function Base.show(io::IO, bf::BenchmarkFunction)
    print(io, bf.name)
end
