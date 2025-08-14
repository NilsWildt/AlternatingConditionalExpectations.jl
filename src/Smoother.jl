module Smoothers

using DispatchDoctor
using Statistics
using LinearAlgebra
using TrendDecomposition
using ConcreteStructs

export Smoother, LAS, LASb, LLSS, LLSSb, do_smoothing, loocv, FRSS
export Kernelsmooth, NWKernelsmooth, guess_parameters!
export TrendSmoother

include("Kernelregression/Kernelregression.jl")
using .Kernelregression

"""
Base type for all smoothing algorithms
"""
abstract type Smoother end

"""
Validates and sanitizes input data for smoothing operations
"""
@stable function validate_inputs(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, window::Int64)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have same length"))
    window > 0 || throw(ArgumentError("Window size must be positive"))
    window < length(x) || throw(ArgumentError("Window size must be less than data length"))
    
    # Ensure window is odd
    iseven(window) && (window += 1)
    return Int64((window - 1) ÷ 2)
end

@concrete struct LAS <: Smoother
    window::Int64
    LAS(window::Number) = new(Int64(round(window)))
end

@concrete struct LASb <: Smoother
    window::Int64
    LASb(window::Number) = new(Int64(round(window)))
end

@concrete struct LLSS <: Smoother
    window::Int64
    LLSS(window::Number) = new(Int64(round(window)))
end

@concrete struct LLSSb <: Smoother
    window::Int64
    LLSSb(window::Number) = new(Int64(round(window)))
end

"""
Performs Local Average Smoothing (LAS)
"""
@stable function do_smoothing(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, smoother::V) where {V<:LAS}
    T = promote_type(eltype(x), eltype(y))
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    result = zeros(T, n)
    
    for i in 1:n
        window = max(1, i-k):min(n, i+k)
        result[i] = mean(y[window])
    end
    return result
end

"""
Performs Local Average Smoothing with boundary adjustment (LASb)
"""
@stable function do_smoothing(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, smoother::V) where {V<:LASb}
    T = promote_type(eltype(x), eltype(y))
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    result = Vector{T}(undef, n)
    
    @inbounds for i in 1:n
        left_start = max(1, i-k) - min(0, n-i-k+1)
        right_end = min(i+k, n) + min(0, i-k)
        sum_val = zero(T)
        count = 0
        
        for j in left_start:right_end
            sum_val += y[j]
            count += 1
        end
        
        result[i] = sum_val / count
    end
    return result
end

"""
Helper function for linear regression calculations
"""
@stable function linear_regression(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, xi::Real)
    T = promote_type(eltype(x), eltype(y), typeof(xi))
    xm, ym = mean(x), mean(y)
    C = sum((x .- xm) .* (y .- ym))
    V = sum((x .- xm) .^ 2)
    
    β = V ≈ zero(T) ? zero(T) : T(C / V)
    α = T(ym - β * xm)
    return α + β * T(xi)
end

"""
Performs Local Linear Smoothing (LLSS)
"""
@stable function do_smoothing(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, smoother::V) where {V<:LLSS}
    T = promote_type(eltype(x), eltype(y))
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    result = zeros(T, n)
    
    for i in 1:n
        window = max(1, i-k):min(n, i+k)
        result[i] = linear_regression(x[window], y[window], x[i])
    end
    return result
end

"""
Performs Local Linear Smoothing with boundary adjustment (LLSSb)
"""
@stable function do_smoothing(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, smoother::V) where {V<:LLSSb}
    T = promote_type(eltype(x), eltype(y))
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    result = zeros(T, n)
    
    for i in 1:n
        # Compute asymmetric window with boundary adjustment
        left_k = min(k, i - 1)
        right_k = min(k, n - i)
        
        # Adjust window if necessary
        if left_k < k
            right_k = min(n - i, 2 * k - left_k)
        elseif right_k < k
            left_k = min(i - 1, 2 * k - right_k)
        end
        
        window_indices = (i - left_k):(i + right_k)
        
        if length(window_indices) > 0
            window_x = x[window_indices]
            window_y = y[window_indices]
            result[i] = linear_regression(window_x, window_y, x[i])
        else
            result[i] = y[i]
        end
    end
    return result
end

"""
Performs Leave-One-Out Cross-Validation
"""
@stable function loocv(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, smoother::Smoother)
    T = promote_type(eltype(x), eltype(y))
    k = validate_inputs(x, y, smoother.window)
    n = length(x)
    ysmoothed = do_smoothing(x, y, smoother)
    cv = zeros(T, n)
    
    for i in 1:n
        left = max(1, i-k):i-1
        right = i+1:min(n, i+k)
        window_indices = vcat(collect(left), collect(right))
        
        if !isempty(window_indices)
            window_x = x[window_indices]
            xm = mean(window_x)
            v = var(window_x; corrected=false, mean=xm)
            denom = one(T) - one(T)/(2k) - (v ≈ zero(T) ? zero(T) : (x[i] - xm)^2/v)
            cv[i] = abs(denom) < eps(T) ? zero(T) : (y[i] - ysmoothed[i])/denom
        end
    end
    
    return abs.(cv), ysmoothed
end

@concrete mutable struct FRSS{T} <: Smoother
    initial_Js::Vector{T}
    medium_J::T
    final_J::T
    
    function FRSS{T}(inJs::AbstractVector{T}, medJ::T, finalJ::T) where T<:Real
        new{T}(sort(unique(collect(inJs))), medJ, finalJ)
    end
end

# Constructors for mixed types and generic arrays
FRSS(inJs::AbstractVector{T}, medJ::T, finalJ::T) where T<:Real = FRSS{T}(inJs, medJ, finalJ)
FRSS(inJs, medJ::T, finalJ::T) where T<:Real = FRSS{T}(Vector{T}(inJs), medJ, finalJ)

FRSS() = FRSS{Float64}([0.05,0.2,0.5], 0.2, 0.05)


@stable function perform_initial_smoothing(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, 
                                          smoother::FRSS{T}, Nx::Int, standardsmooth) where T<:Real
    initial_Js_array = repeat(smoother.initial_Js', Nx, 1)
    initial_smooth = zeros(T, Nx, length(smoother.initial_Js))
    initial_cv_residuals = zeros(T, Nx, length(smoother.initial_Js))
    
    @inbounds for (i, J) in enumerate(Nx .* smoother.initial_Js)
        J_int = max(3, Int(round(J)))  # Ensure window is at least 3
        lin_smoother = standardsmooth(J_int)
        initial_cv_residuals[:, i], initial_smooth[:, i] = loocv(x, y, lin_smoother)
    end
    
    return initial_Js_array, initial_smooth, initial_cv_residuals
end

@stable function smooth_residuals(x::AbstractArray{<:Real}, residuals::AbstractArray{<:Real}, 
                                 smoother::FRSS{T}, Nx::Int, standardsmooth) where T<:Real
    J_int = max(3, Int(round(smoother.medium_J * Nx)))
    lin_smoother = standardsmooth(J_int)
    return mapslices(col -> do_smoothing(x, col, lin_smoother), 
                    residuals, dims=1)
end

@stable function interpolate_smooth(x::AbstractArray{<:Real}, initial_smooth::Matrix{T}, 
                                   smoothed_best_Js::AbstractArray{<:Real}, 
                                   smoother::FRSS{T}, Nx::Int) where T<:Real
    interpolated_smooth = zeros(T, Nx)
    
    @inbounds for i in 1:Nx
        # Find the two closest initial J values
        j1 = 1
        j2 = length(smoother.initial_Js)
        
        for j in 2:length(smoother.initial_Js)
            if abs(smoothed_best_Js[i] - smoother.initial_Js[j]) < 
               abs(smoothed_best_Js[i] - smoother.initial_Js[j1])
                j2 = j1
                j1 = j
            elseif abs(smoothed_best_Js[i] - smoother.initial_Js[j]) < 
                   abs(smoothed_best_Js[i] - smoother.initial_Js[j2])
                j2 = j
            end
        end
        
        # Ensure j1 < j2
        if j1 > j2
            j1, j2 = j2, j1
        end
        
        # Linear interpolation
        if smoother.initial_Js[j1] ≈ smoother.initial_Js[j2]
            interpolated_smooth[i] = initial_smooth[i, j1]
        else
            interpolated_smooth[i] = (initial_smooth[i, j2] - initial_smooth[i, j1]) / 
                                    (smoother.initial_Js[j2] - smoother.initial_Js[j1]) * 
                                    (smoothed_best_Js[i] - smoother.initial_Js[j1]) + 
                                    initial_smooth[i, j1]
        end
    end
    
    return interpolated_smooth
end

@stable function do_smoothing(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, 
                             smoother::FRSS{T}) where T<:Real
    Nx = length(x)
    standardsmooth = LLSSb

    # Step 1: Initial smoothing
    initial_Js_array, initial_smooth, initial_cv_residuals = 
        perform_initial_smoothing(x, y, smoother, Nx, standardsmooth)

    # Step 2: Smooth residual curves
    initial_residuals_smoothed = 
        smooth_residuals(x, initial_cv_residuals, smoother, Nx, standardsmooth)

    # Step 3: Get best Js
    best_J_indices = [argmin(initial_residuals_smoothed[i, :]) for i in 1:Nx]
    best_Js = T[initial_Js_array[i, best_J_indices[i]] for i in 1:Nx]

    # Step 4: Smooth best Js
    J_int = max(3, Int(round(smoother.medium_J * Nx)))
    lin_smoother = standardsmooth(J_int)
    smoothed_best_Js = do_smoothing(x, best_Js, lin_smoother)
    clamp!(smoothed_best_Js, smoother.initial_Js[1], smoother.initial_Js[end])

    # Step 5: Interpolate
    interpolated_smooth = interpolate_smooth(x, initial_smooth, smoothed_best_Js, smoother, Nx)

    # Step 6: Final smoothing
    J_int = max(3, Int(round(smoother.final_J * Nx)))
    lin_smoother = standardsmooth(J_int)
    final_smooth = do_smoothing(x, interpolated_smooth, lin_smoother)
    
    return final_smooth
end

Base.String(frss::FRSS) = "Smoothed_FRSS"

# In case we have several smoothers, apply them in order
@stable function do_smoothing(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, 
                             smoothers::Array{<:Smoother})
    result = copy(y)
    for sm in smoothers
        result = do_smoothing(x, result, sm)
    end
    return result
end

# More specific method for concrete array types to improve type inference
@stable function do_smoothing(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, 
                             smoothers::Vector{S}) where S<:Smoother
    result = copy(y)
    for sm in smoothers
        result = do_smoothing(x, result, sm)
    end
    return result
end

# Handle mixed smoother arrays with explicit return type
@stable function do_smoothing(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, 
                             smoothers::Vector{<:Smoother})
    T = promote_type(eltype(x), eltype(y))
    result = copy(y)
    for sm in smoothers
        result = do_smoothing(x, result, sm)
    end
    return result
end

@stable function do_smoothing!(out::AbstractArray{<:Real}, x::AbstractArray{<:Real}, 
                               y::AbstractArray{<:Real}, smoothers::Array{<:Smoother})
    temp = copy(y)
    for sm in smoothers
        out .= do_smoothing(x, temp, sm)
        temp .= out
    end
end

# More specific method for concrete array types to improve type inference
@stable function do_smoothing!(out::AbstractArray{<:Real}, x::AbstractArray{<:Real}, 
                               y::AbstractArray{<:Real}, smoothers::Vector{S}) where S<:Smoother
    temp = copy(y)
    for sm in smoothers
        out .= do_smoothing(x, temp, sm)
        temp .= out
    end
end

# Kernel smoothing structures
@concrete mutable struct Kernelsmooth{T<:Real, K<:Kernelregression.SKernel} <: Smoother
    reg::T
    smoothk::K
end

# Convenience constructors
Kernelsmooth(reg::T, smoothk::K) where {T<:Real, K<:Kernelregression.SKernel} = Kernelsmooth{T,K}(reg, smoothk)

@stable function do_smoothing(x::AbstractVector{<:Real}, y::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}}, 
                             smoother::Kernelsmooth{T,K}) where {T<:Real, K<:Kernelregression.SKernel}
    T2 = promote_type(eltype(x), eltype(y), T)
    x_array = Array{T2,1}(x)
    y_array = Array{T2,1}(vec(y))
    Nx = length(x)

    xeval = collect(LinRange(x[1], x[end], Nx))
    kInterpolant = Kernelregression.get_kernel_interpolant(x_array, y_array, smoother.smoothk, smoother.reg)
    return kInterpolant(xeval)
end

@stable function do_smoothing!(out::AbstractArray{<:Real}, x::AbstractVector{<:Real}, 
                               y::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}}, 
                               smoother::Kernelsmooth{T,K}) where {T<:Real, K<:Kernelregression.SKernel}
    T2 = promote_type(eltype(x), eltype(y), T)
    x_array = Array{T2,1}(x)
    y_array = Array{T2,1}(vec(y))
    Nx = length(x)

    xeval = collect(LinRange(x[1], x[end], Nx))
    kInterpolant = Kernelregression.get_kernel_interpolant(x_array, y_array, smoother.smoothk, smoother.reg)
    out .= kInterpolant(xeval)
end

@stable function Base.String(ks::Kernelsmooth)
    try
        string("Smoothed_Mercer_", String(ks.smoothk), "($(ks.smoothk.σ))")
    catch
        string("Smoothed_Mercer_", String(ks.smoothk))
    end
end

@concrete mutable struct NWKernelsmooth{T<:Real, K<:Kernelregression.SKernel} <: Smoother
    smoothk::K
    
    NWKernelsmooth{T,K}(smoothk::K) where {T<:Real, K<:Kernelregression.SKernel} = new{T,K}(smoothk)
end

# Constructors for type inference
NWKernelsmooth(smoothk::K) where {K<:Kernelregression.SKernel} = NWKernelsmooth{Float64,K}(smoothk)
NWKernelsmooth{T}(smoothk::K) where {T<:Real, K<:Kernelregression.SKernel} = NWKernelsmooth{T,K}(smoothk)

@stable function do_smoothing(x::AbstractVector{<:Real}, y::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}}, 
                             smoother::NWKernelsmooth{T,K}) where {T<:Real, K<:Kernelregression.SKernel}
    T2 = promote_type(eltype(x), eltype(y), T)
    Nx = length(x)
    y_vec = vec(y)
    
    m(a, b) = Kernelregression.evalKernel(smoother.smoothk, a - b)
    retval = zeros(T2, Nx)
    
    @inbounds for i in 1:Nx
        kernel_weights = T2[m(x[i], x[j]) for j in 1:Nx]
        weight_sum = sum(kernel_weights)
        if weight_sum > zero(T2)
            normalized_weights = kernel_weights ./ weight_sum
            retval[i] = sum(normalized_weights .* y_vec)
        else
            retval[i] = y_vec[i]
        end
    end
    return retval
end

@stable function do_smoothing!(out::AbstractArray{<:Real}, x::AbstractVector{<:Real}, 
                               y::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}}, 
                               smoother::NWKernelsmooth{T,K}) where {T<:Real, K<:Kernelregression.SKernel}
    T2 = promote_type(eltype(x), eltype(y), T, eltype(out))
    Nx = length(x)
    y_vec = vec(y)
    
    m(a, b) = Kernelregression.evalKernel(smoother.smoothk, a - b)
    
    @inbounds for i in 1:Nx
        kernel_weights = T2[m(x[i], x[j]) for j in 1:Nx]
        weight_sum = sum(kernel_weights)
        if weight_sum > zero(T2)
            normalized_weights = kernel_weights ./ weight_sum
            out[i] = sum(normalized_weights .* y_vec)
        else
            out[i] = y_vec[i]
        end
    end
end

@stable function Base.String(ks::NWKernelsmooth)
    try
        string("Smoothed_NW_", String(ks.smoothk), "($(round(ks.smoothk.σ; digits=4)))")
    catch
        string("Smoothed_NW_", String(ks.smoothk))
    end
end

@stable function guess_parameters!(Y::Array{<:Real}, mysmoother::NWKernelsmooth{T}) where T<:Real
    N = length(Y)
    try
        iqr_val = quantile(Y, T(0.75)) - quantile(Y, T(0.25))
        mysmoother.smoothk.σ = T(0.9) * minimum([std(Y), iqr_val / T(1.34)]) * N^(T(-1/5))
        println("We're doing it with σ = $(mysmoother.smoothk.σ)")
    catch e
        println("Error computing kernel bandwidth: $e")
    end
end

# TrendDecomposition-based smoother
@concrete struct TrendSmoother{T<:Real} <: Smoother
    λ::T  # Smoothing parameter for HP filter
    window::Int64  # Window size for moving averages
    weights::Union{Vector{T}, Nothing}  # Optional weights for weighted moving average
    centered::Bool  # Whether to center the moving average
    offset::Int64  # Offset for moving average
    smoother_type::Symbol  # :hp, :ma, :ma_weighted
end

# Constructors for different trend smoothing methods
function TrendSmoother(λ::T; smoother_type=:hp) where T<:Real
    smoother_type == :hp || throw(ArgumentError("For HP filter, only λ parameter is needed"))
    return TrendSmoother{T}(λ, 0, nothing, true, 0, smoother_type)
end

function TrendSmoother(window::Int; weights=nothing, centered=true, offset=0, smoother_type=:ma)
    if smoother_type == :ma_weighted && isnothing(weights)
        throw(ArgumentError("Weights must be provided for weighted moving average"))
    end
    if smoother_type == :ma_weighted && length(weights) != window
        throw(ArgumentError("Number of weights must match window size"))
    end
    
    T = isnothing(weights) ? Float64 : eltype(weights)
    return TrendSmoother{T}(0.0, window, weights, centered, offset, smoother_type)
end

@stable function do_smoothing(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, 
                             smoother::TrendSmoother{T}) where T<:Real
    y_vec = collect(vec(y))
    if smoother.smoother_type == :hp
        return hpFilter(y_vec, smoother.λ)
    elseif smoother.smoother_type == :ma
        if isnothing(smoother.weights)
            return rollingAverage(y_vec, smoother.window, centered=smoother.centered, offset=smoother.offset)
        else
            return rollingAverage(y_vec, smoother.window, smoother.weights, centered=smoother.centered, offset=smoother.offset)
        end
    elseif smoother.smoother_type == :ma_weighted
        return rollingAverage(y_vec, smoother.window, smoother.weights, centered=smoother.centered, offset=smoother.offset)
    else
        throw(ArgumentError("Unknown smoother_type=$(smoother.smoother_type)"))
    end
end

# In-place versions
@stable function do_smoothing!(out::AbstractArray{<:Real}, x::AbstractArray{<:Real}, 
                               y::AbstractArray{<:Real}, smoother::TrendSmoother{T}) where T<:Real
    result = do_smoothing(x, y, smoother)
    out .= result
end

Base.String(ts::TrendSmoother) = "TrendSmoother($(ts.smoother_type))"

# Display helpers for table output
Base.String(l::LAS) = "LAS($(l.window))"
Base.String(l::LASb) = "LASb($(l.window))"
Base.String(l::LLSS) = "LLSS($(l.window))"
Base.String(l::LLSSb) = "LLSSb($(l.window))"

end # module