module Smoothers
export Smoother, LAS, LASb, LLSS, LLSSb, do_smoothing, loocv, FRSS
using Statistics, LinearAlgebra
include("Kernelregression/Kernelregression.jl")
using .Kernelregression
"""
Base type for all smoothing algorithms
"""
abstract type Smoother end

"""
Validates and sanitizes input data for smoothing operations
"""
function validate_inputs(x::AbstractArray{T}, y::AbstractArray{T}, window::Int64) where T<:Real
    length(x) == length(y) || throw(DimensionMismatch("x and y must have same length"))
    window > 0 || throw(ArgumentError("Window size must be positive"))
    window < length(x) || throw(ArgumentError("Window size must be less than data length"))
    
    # Ensure window is odd
    iseven(window) && (window += 1)
    return Int64((window - 1) ÷ 2)
end

struct LAS <: Smoother
    window::Int64
    LAS(window::Number) = new(Int64(round(window)))
end

struct LASb <: Smoother
    window::Int64
    LASb(window::Number) = new(Int64(round(window)))
end

struct LLSS <: Smoother
    window::Int64
    LLSS(window::Number) = new(Int64(round(window)))
end

struct LLSSb <: Smoother
    window::Int64
    LLSSb(window::Number) = new(Int64(round(window)))
end
"""
Performs Local Average Smoothing (LAS)
"""
function do_smoothing(x::AbstractArray{T}, y::AbstractArray{T}, smoother::V) where {T<:Real, V<:LAS}
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    result = zeros(Float64, n)
    
    for i in 1:n
        window = max(1, i-k):min(n, i+k)
        result[i] = mean(y[window])
    end
    return result
end

"""
Performs Local Average Smoothing with boundary adjustment (LASb)
"""
function do_smoothing(x::AbstractArray{T}, y::AbstractArray{T}, smoother::V) where {T<:Real, V<:LASb}
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    result = zeros(Float64, n)
    
    for i in 1:n
        left = max(1, i-k)-min(0, n-i-k+1):i
        right = i+1:min(i+k, n)+min(0, i-k)
        result[i] = mean(vcat(y[left], y[right]))
    end
    return result
end

"""
Helper function for linear regression calculations
"""
function linear_regression(x::AbstractArray{T}, y::AbstractArray{T}, xi::T) where T<:Real
    xm, ym = mean(x), mean(y)
    C = sum((x .- xm) .* (y .- ym))
    V = sum((x .- xm) .^ 2)
    
    β = V ≈ zero(T) ? zero(T) : C / V
    α = ym - β * xm
    return α + β * xi
end

"""
Performs Local Linear Smoothing (LLSS)
"""
function do_smoothing(x::AbstractArray{T}, y::AbstractArray{T}, smoother::V) where {T<:Real, V<:LLSS}
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
function do_smoothing(x::AbstractArray{T}, y::AbstractArray{T}, smoother::V) where {T<:Real, V<:LLSSb}
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    result = zeros(T, n)
    
    for i in 1:n
        left = max(1, i-k)-min(0, n-i-k+1):i-1
        right = i:min(i+k, n)+min(0, i-k)
        window_x = vcat(x[left], x[right])
        window_y = vcat(y[left], y[right])
        result[i] = linear_regression(window_x, window_y, x[i])
    end
    return result
end

"""
Performs Leave-One-Out Cross-Validation
"""
function loocv(x::AbstractArray{T}, y::AbstractArray{T}, smoother::Smoother) where {T<:Real}
    k = validate_inputs(x, y, smoother.window)
    n = length(x)
    ysmoothed = do_smoothing(x, y, smoother)
    cv = zeros(Float64, n)
    
    for i in 1:n
        left = max(1, i-k):i-1
        right = i+1:min(n, i+k)
        window = vcat(x[left], x[right])
        
        if !isempty(window)
            xm = mean(window)
            v = var(window; corrected=false, mean=xm)
            denom = (1.0 - 1.0/(2k) - (v ≈ 0 ? 0.0 : (x[i] - xm)/v))
            cv[i] = abs(denom) < eps() ? 0.0 : (y[i] - ysmoothed[i])/denom
        end
    end
    
    return abs.(cv), ysmoothed
end


mutable struct FRSS <: Smoother
    initial_Js::AbstractArray
    medium_J::Float64
    final_J::Float64
    
    function FRSS(inJs, medJ, finalJ)
        new(sort(unique(inJs)), medJ, finalJ)
    end
end

function perform_initial_smoothing(x::T, y::T, smoother::FRSS, Nx::Int, standardsmooth) where {T<:AbstractVecOrMat}
    initial_Js_array = repeat(smoother.initial_Js', Nx, 1)
    initial_smooth = zeros(Float64, Nx, length(smoother.initial_Js))
    initial_cv_residuals = zeros(Float64, Nx, length(smoother.initial_Js))
    
    @inbounds for (i, J) in enumerate(Nx .* smoother.initial_Js)
        lin_smoother = standardsmooth(J)
        initial_cv_residuals[:, i], initial_smooth[:, i] = loocv(x, y, lin_smoother)
    end
    
    return initial_Js_array, initial_smooth, initial_cv_residuals
end

function smooth_residuals(x::T, initial_cv_residuals::Matrix{Float64}, 
                         smoother::FRSS, Nx::Int, standardsmooth) where {T<:AbstractVecOrMat}
    initial_residuals_smoothed = zeros(Float64, size(initial_cv_residuals))
    lin_smoother = standardsmooth(smoother.medium_J * Nx)
    
    @inbounds @simd for i in 1:size(initial_cv_residuals, 2)
        initial_residuals_smoothed[:, i] = do_smoothing(x, initial_cv_residuals[:, i], lin_smoother)
    end
    
    return initial_residuals_smoothed
end

function interpolate_smooth(x::T, initial_smooth::Matrix{Float64}, smoothed_best_Js::AbstractArray, 
                          smoother::FRSS, Nx::Int) where {T<:AbstractVecOrMat}
    interpolated_smooth = zeros(Float64, Nx, 1)
    
    @inbounds @simd for i in 1:Nx
        j1 = abs(smoothed_best_Js[i] - smoother.initial_Js[1]) < 
             abs(smoothed_best_Js[i] - smoother.initial_Js[2]) ? 1 : 2
        j2 = 3
        
        interpolated_smooth[i] = (initial_smooth[i, j1] - initial_smooth[i, j2]) / 
                                (smoother.initial_Js[j1] - smoother.initial_Js[j2]) * 
                                (smoothed_best_Js[i] - smoother.initial_Js[j2]) + 
                                initial_smooth[i, j2]
    end
    
    return interpolated_smooth
end

function do_smoothing(x::T, y::T, smoother::V) where {T<:AbstractVecOrMat, V<:FRSS}
    Nx = length(x)
    standardsmooth = LLSSb

    # Step 1: Initial smoothing
    initial_Js_array, initial_smooth, initial_cv_residuals = 
        perform_initial_smoothing(x, y, smoother, Nx, standardsmooth)

    # Step 2: Smooth residual curves
    initial_residuals_smoothed = 
        smooth_residuals(x, initial_cv_residuals, smoother, Nx, standardsmooth)

    # Step 3: Get best Js
    best_Js = initial_Js_array[argmin(initial_residuals_smoothed, dims=2)]

    # Step 4: Smooth best Js
    lin_smoother = standardsmooth(smoother.medium_J * Nx)
    smoothed_best_Js = do_smoothing(x, best_Js, lin_smoother)
    clamp!(Array(smoothed_best_Js), smoother.initial_Js[1], smoother.initial_Js[end])

    # Step 5: Interpolate
    interpolated_smooth = interpolate_smooth(x, initial_smooth, smoothed_best_Js, smoother, Nx)

    # Step 6: Final smoothing
    lin_smoother = standardsmooth(smoother.final_J * Nx)
    final_smooth = do_smoothing(x, interpolated_smooth, lin_smoother)
    
    return final_smooth
end

Base.String(frss::FRSS) = "Smoothed_FRSS"



# In case, we gave it several smoothers, do them in this order every time
function do_smoothing(x::AbstractVecOrMat, y::AbstractVecOrMat, smoothers::Array{<:Smoother})
    for sm in smoothers::Vector{Smoother}
        y = do_smoothing(x, y, sm)
    end
    return y
end

function do_smoothing!(out, x::AbstractVecOrMat, y::AbstractVecOrMat, smoothers::Array{<:Smoother})
    for sm in smoothers::Vector{Smoother}
        do_smoothing!(out, x, y, sm)
        y .= out
    end
end


mutable struct Kernelsmooth <: Smoother
    reg::Float64
    smoothk::SKernel
end


function do_smoothing(x::Vector{Float64}, y::VecOrMat{Float64}, smoother::Kernelsmooth)
    # x, y =   _sanitizeinput(x, y)
    x = Array{Float64,1}(x)
    y = Array{Float64,1}(y)
    Nx = length(x)

    xeval = collect(LinRange(x[1], x[end], Nx))
    kInterpolant = Kernelregression.get_kernel_interpolant(x, y, smoother.smoothk, smoother.reg)
    return kInterpolant(xeval)
end



function do_smoothing!(out, x::Vector{Float64}, y::VecOrMat{Float64}, smoother::Kernelsmooth)
    # x, y =   _sanitizeinput(x, y)
    x = Array{Float64,1}(x)
    y = Array{Float64,1}(y)
    Nx = length(x)

    xeval = collect(LinRange(x[1], x[end], Nx))
    kInterpolant = Kernelregression.get_kernel_interpolant(x, y, smoother.smoothk, smoother.reg)
    out .= kInterpolant(xeval)
end

function Base.String(ks::Kernelsmooth)
    try
        string("Smoothed_Mercer_", String(ks.smoothk), "($(ks.smoothk.σ))")
    catch
        string("Smoothed_Mercer_", String(ks.smoothk))
    end
end





mutable struct NWKernelsmooth <: Smoother
    smoothk::SKernel
end

function do_smoothing(x::Vector{Float64}, y::VecOrMat{Float64}, smoother::NWKernelsmooth)
    # x, y =   _sanitizeinput(x, y)
    Nx = length(x)

    m(a, b) = Kernelregression.evalKernel(smoother.smoothk, a .- b)
    retval = zeros(size(x))
    @inbounds @simd for i in 1:Nx
        tmp = m(view(x,i), x)
        retval[i] = mymean((tmp ./ mean(tmp)) .* y)
    end
    return retval
end

function do_smoothing!(out,x::Vector{Float64}, y::VecOrMat{Float64}, smoother::NWKernelsmooth)
    # x, y =   _sanitizeinput(x, y)
    Nx = length(x)

    m(a, b) = Kernelregression.evalKernel(smoother.smoothk, a .- b)
    @inbounds @simd for i in 1:Nx
        tmp = m(x[i], x)
        out[i] = mean((tmp ./ mean(tmp)) .* y)
    end
end


function Base.String(ks::NWKernelsmooth)
    try
        string("Smoothed_NW_", String(ks.smoothk), "($(round(ks.smoothk.σ;digits = 4)))")
    catch
        string("Smoothed_NW_", String(ks.smoothk))
    end
end


function guess_parameters!(Y::Array{Float64}, mysmoother::NWKernelsmooth)
    N = length(Y)
    try
        smoother.smoothk.σ = 0.9 * minimum(std(Y), iqr(Y) / 1.34) * N^(-1 / 5) # ((4 * std(Y)^5) / (3 * N)).^(1 / 5)
        println("We're doing it with k = $(mysmoother.k)")
    catch e
        println("We don't have a k, as we're probably using kernels.")
    end
end


end # module