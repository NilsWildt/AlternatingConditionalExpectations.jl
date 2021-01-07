using Random
using DocStringExtensions
using Documenter



"""
Heaviside
===

Calculate the heaviside function.

Use: 
$(TYPEDSIGNATURES)
"""
function heaviside(x)
    return @.  0.5 * (sign(x) + 1.0)
end


"""
Normal sampler 
===

See above.

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector


σ: std and μ: mean 

seed: (standard = 0, which refers to no given seed) for MersenneTwister
"""
function normal_sample(numSamples::Int64, numDim::Int64, μ::Float64 = 0.0, σ::Float64 = 1.0, seed::Int64 = 0)
    dims = (numSamples, numDim)
    rng = seed == 0 ? MersenneTwister() : MersenneTwister(seed)
    return  σ .* randn(rng,  Float64, dims) .+ μ
end



"""
Uniform sampler 
===

See above.

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

bounds: Tuple(lower_bound, upper_bound)

seed: (standard = 0, which refers to no given seed) for MersenneTwister
"""
function uniform_sample(numSamples::Int64, numDim::Int64,  σ_noise::Float64, seed::Int64 = 0, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
    lb, ub = bounds
    dims = (numSamples, numDim)
    rng = seed == 0 ? MersenneTwister() : MersenneTwister(seed)
    return abs(ub - lb) .* (rand(rng,  Float64, dims)) .+ lb
end

"""
$(FUNCTIONNAME)
===

Returns the correct sample, dependent on "samplingmethod"

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

samplingmethod: 1. "normal" 2. "uniform"

seed: (standard = 0, which refers to no given seed) for MersenneTwister

bounds
"""
function get_sample(numSamples::Int64, numDim::Int64, samplingmethod::String, σ_x::Float64, σ_noise::Float64, seed::Int64 = 0, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
         if samplingmethod == "uniform"
        X = uniform_sample(numSamples, numDim, σ_x, seed, bounds)
    elseif samplingmethod == "normal"
        X = normal_sample(numSamples, numDim, 0.0, σ_x, seed)
    else 
        @warn "You didn't provide a sampling method. Using normal distributed samples now."
           X = normal_sample(numSamples, numDim, seed)
    end
    return X
end

"""
$(FUNCTIONNAME)
===

$(raw"""
f_{b1}(x_1)= x_2 = \exp{(x_1^3 + \varepsilon)})
""")

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

samplingmethod: 1. "normal" 2. "uniform"

seed: (standard = 0, which refers to no given seed) for MersenneTwister

bounds
"""
function f_b1(numSamples::Int64, numDim::Int64, samplingmethod::String, σ_x::Float64, σ_noise::Float64, seed::Int64 = 0, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
        lb, ub = bounds
   X = get_sample(numSamples, numDim, samplingmethod, σ_x, σ_noise, seed, bounds)
    Y =  exp.(X.^3 + normal_sample(numSamples, numDim, 0.0, σ_noise, seed + 1))
    return X, Y
end


"""
$(FUNCTIONNAME)
===

$(raw"""
f_{b2}(x_1)= x_2 = \exp{(\sin{(x_1)} + \varepsilon)}
""")

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

samplingmethod: 1. "normal" 2. "uniform"

seed: (standard = 0, which refers to no given seed) for MersenneTwister

bounds
"""
function f_b2(numSamples::Int64, numDim::Int64, samplingmethod::String, σ_x::Float64, σ_noise::Float64, seed::Int64 = 0, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
     lb, ub = bounds
     X = get_sample(numSamples, numDim, samplingmethod, σ_x, σ_noise, seed, bounds)
    Y =  exp.(sin.(X) +  normal_sample(numSamples, numDim, 0.0, σ_x, seed + 1))
    return X, Y
end


"""
$(FUNCTIONNAME)
===

$(raw"""
f_{b3}(x_1)= x_2 = \sigma(x_2)
""")

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

samplingmethod: 1. "normal" 2. "uniform"

seed: (standard = 0, which refers to no given seed) for MersenneTwister

bounds
"""
function f_b3(numSamples::Int64, numDim::Int64, samplingmethod::String, σ_x::Float64, σ_noise::Float64,  seed::Int64 = 0, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
         lb, ub = bounds
     X = get_sample(numSamples, numDim, samplingmethod, σ_x, σ_noise, seed, bounds)
    Y = heaviside.(X) + normal_sample(numSamples, numDim, 0.0, σ_x, seed + 1)
    return X, Y
end

"""
$(FUNCTIONNAME)
===

$(raw"""
f_{b4}(x_1)= x_2 = \log{(\log{(x_2)})}
""")

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

samplingmethod: 1. "normal" 2. "uniform"

seed: (standard = 0, which refers to no given seed) for MersenneTwister

bounds
"""
function f_b4(numSamples::Int64, numDim::Int64,  samplingmethod::String, σ_x::Float64, σ_noise::Float64, seed::Int64 = 0, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
        lb, ub = bounds
     X = get_sample(numSamples, numDim, samplingmethod, σ_x, σ_noise, seed, bounds)
    Y = log.(abs.(X + normal_sample(numSamples, numDim, 0.0, σ_x, seed + 1)))
    return X, Y
end



"""
$(FUNCTIONNAME)
===

$(raw"""
f_{toy} = X + noise
""")

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

samplingmethod: 1. "normal" 2. "uniform"

seed: (standard = 0, which refers to no given seed) for MersenneTwister

bounds
"""
function f_toy(numSamples::Int64, numDim::Int64,  samplingmethod::String, σ_x::Float64, σ_noise::Float64, seed::Int64 = 0, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
        lb, ub = bounds
     X = get_sample(numSamples, numDim, samplingmethod, σ_x, σ_noise, seed, bounds)
    Y = X .+ normal_sample(numSamples, numDim, 0.0, σ_x, seed + 1)
    return X, Y
end