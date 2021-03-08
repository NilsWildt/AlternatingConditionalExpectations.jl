include("Kernelregression/Kernelregression.jl")
using .Kernelregression
using Interpolations 
using DocStringExtensions
Base.Experimental.@optlevel 3   
mutable struct LAS <: Smoother
    k::Int64
    function LAS(k) 
        new(_sanitize_k(k))
    end
end

# @deprecate function guess_k!(mysmoother::Smoother, N::Int64)
#     guess_parameter!(mysmoother, N)
# end

function guess_parameters!(mysmoother::Smoother, N::Int64)
    try
        mysmoother.k = Int64(floor((N + 2) / 4))
        println("We're doing it with k = $(mysmoother.k)")
    catch e
        println("We don't have a k, as we're probably using kernels.")
    end
end

@fastmath function do_smoothing(x::T, y::T, smoother::LAS)::T where {T <: AbstractArray{Float64}}
    k = smoother.k
    # Nx = length(x)
    Ny = length(y)
    LASvals =  zeros(Float64, Ny)
    c = 1
    @inbounds @simd for i in 1:Ny
        LASvals[i] =  sum(@views  y[max(i - k, 1):min(i + k, Ny)])
        c = c + 1
    end
    return LASvals ./ c
end

Base.String(k::LAS) = "Smoothed_LAS($(2 * k.k + 1))"

mutable struct LASb <: Smoother
    k::Int64
end

@fastmath function do_smoothing(x::T, y::T, smoother::LASb)::T where {T <: AbstractArray{Float64}}
    k = smoother.k
    Ny = length(y)
    LASbvals =  zeros(Float64, (Ny))
    @inbounds  for i in eachindex(y)
        ind_low =  max(i - k, 1) - min(0, Ny - i - k + 1):i 
        ind_high =  i + 1:min(i + k, Ny) + min(0, i - k)
        LASbvals[i] =    mean(@views vcat(y[ind_low],  y[ind_high]))
    end
    return LASbvals
end

Base.String(k::LASb) = "Smoothed_LASb($(2 * k.k + 1))"

###

mutable struct LLSS <: Smoother
    k::Int64
    function LLSS(k) 
        new(_sanitize_k(k))
    end
end

function do_smoothing_old(x::T, y::T, smoother::LLSS)::T where {T <: AbstractArray{Float64}}
    k = smoother.k
    # if !presorted
        # x, y =   _sanitizeinput(x, y)
    # end
    Nx = length(x)
    Ny = length(y)
    LLSS_values =  zeros(Float64, Nx)
    
    C =  zeros(Float64, Nx)
    V =  zeros(Float64, Nx)
    α =  zeros(Float64, Nx)
    β =  zeros(Float64, Nx)
    
    # Start at the leftmost point...
    ind = 2:k + 1 
    
    xmean = mean(x[ind])
    ymean = mean(y[ind])
    C[1] = sum((x[ind] .- xmean) .* (y[ind] .- ymean))
    V[1] = sum((x[ind] .- xmean).^2)
    if V[1] == 0.0
        β[1] = 0.0
    else
        β[1] = C[1] / V[1]
    end
    α[1] = -β[1] * xmean + ymean
    LLSS_values[1] = α[1] + β[1] * x[1]
    @inbounds @simd  for i in 2:Nx 
        if (i <= k)# First we are adding points until we have 2k+1 points in total
            indlow = 1:i - 1
            indhigh = i:k + i 
            xind = vcat(x[indlow], x[indhigh])
            yind = vcat(y[indlow], y[indhigh])
            m = length(ind)
            xmean = mean(xind)
            ymean = mean(yind)
            C[i] = sum((xind .- xmean) .* (yind .- ymean))
            V[i] = sum((xind .- xmean).^2)
            if V[i] == 0.0
                @warn "Probably some points are duplicated? Denominator becomes zero." i k 
                β[i] = 0.0
            else
                β[i] = C[i] / V[i]
            end
            α[i] = -β[i] * xmean + ymean
        elseif (i + k > Nx - 1) # Case "am Ende"
            
            indlow = i - k:i - 1
            indhigh = i:Nx
            xind = @views vcat(x[indlow], x[indhigh])
            yind = @views vcat(y[indlow], y[indhigh])
            m = length(xind)
            xmean = mean(xind)
            ymean = mean(yind)
            C[i] = sum((xind .- xmean) .* (yind .- ymean))
            V[i] = sum((xind .- xmean).^2)
            if V[i] == 0.0
                @warn "Probably some points are duplicated? Denominator becomes zero." i k
                β[i] = 0.0
            else
                β[i] = C[i] / V[i]
            end
            α[i] = -β[i] * xmean + ymean
        else # Case einer dayzu, einer weg
            indlow = i - k:i - 1
            indhigh = i:i + k
            xind = @views vcat(x[indlow], x[indhigh])
            yind = @views vcat(y[indlow], y[indhigh])
            xmean = mean(xind)
            ymean = mean(yind)
            C[i] = sum((xind .- xmean) .* (yind .- ymean))
            V[i] = sum((xind .- xmean).^2)
            if V[i] == 0.0

                @warn "Probably some points are duplicated? Denominator becomes zero." i k
                β[i] = 0.0
            else
                β[i] = @views C[i] / V[i]
            end
            α[i] = -β[i] * xmean + ymean
        end
        LLSS_values[i] = @views α[i] + β[i] * x[i]
    end
    return  LLSS_values
end

@fastmath function do_smoothing(x::T, y::T, smoother::LLSS)::T where {T <: AbstractArray{Float64}}
    k = smoother.k
    Ny = length(y)
    LLSS_values =  zeros(Float64, Ny)
    
    C = 0.0
    V = 0.0
   
        # Start at the leftmost point...
    @inbounds @simd for i = Base.OneTo(Ny)
        ind = max(i - k, 1):min(i + k, Ny)
        Nind = length(ind)
        xm = mean(x[ind])
        ym = mean(y[ind])
        C = 0.0
        V = 0.0
        @inbounds   for si in ind
            C = C .+ (x[si] .- xm) * (y[si] .- ym)
            V = V .+ (x[si] .- xm).^2
        end
        β = C / V
        α = ym .- β' * xm
        # @info "dbg" size(LLSS_values) α  β xm  LLSS_values[i]  (α .+ β .* x[i]) 
        LLSS_values[i] = α .+ β .* x[i]
    end

    return  LLSS_values
end


# Smoothing in the smoother.k*2+1 box but calculating abs(y-smoothedvals) plus do LOOCV.
function loocv(x::T, y::T, smoother::LLSS)::T where {T <: AbstractArray{Float64}}
    k = smoother.k
    # if !presorted
        # x, y =   _sanitizeinput(x, y)
    # end
    Nx = length(x)
    Ny = length(y)
    ysmoothed = do_smoothing(x, y, smoother, true)
    # preallocate
    cv =  zeros(Float64, (Nx,))
    @inbounds @simd for i in 1:Nx
        ind = max(i - k, 1):min(i + k, Nx) # actually take 2k+1 values...
        xmean = mean(x[ind])
        denom = (1.0 - 1.0 / (2 * k) - (x[i] - xmean) / var(x[ind]; corrected = false, mean = xmean) ) 

        cv[i]  = (y[i] - ysmoothed[i] ) / denom# As in  A VARUBLE SPAN SMOOTHER by Friedman 1984
    end
    return abs.(cv), ysmoothed
end



Base.String(k::LLSS) = "Smoothed_LLSS($(2 * k.k + 1))"

####

mutable struct LLSSb <: Smoother
    k::Int64
    function LLSSb(k) 
        new(_sanitize_k(k))
    end
end


function do_smoothing(x::T, y::T, smoother::LLSSb)::T where {T <: AbstractArray{Float64}}
    k = smoother.k
 
    # if !presorted
        # x, y =   _sanitizeinput(x, y)
    # end
    Ny = length(y)
    LLSSbvals =  zeros(Float64, Ny)
    
    @fastmath @inbounds @simd  for i in 1:Ny # @inbounds @simd 
        indlow = max(i - k, 1) - min(0, Ny - i - k + 1):i - 1
        indhigh = i:min(i + k, Ny) + min(0, i - k)
        xind = vcat(x[indlow], x[indhigh])
        yind = vcat(y[indlow], y[indhigh])

        xmean = mean(xind)
        ymean = mean(yind)
        C = sum((xind .- xmean) .* (yind .- ymean))
        V = sum((xind .- xmean).^2)
        β = C / V
        α = -β * xmean + ymean
        LLSSbvals[i] = α + β * x[i]
    end
    return   LLSSbvals
end


function do_smoothing_slow(x::T, y::T, smoother::LLSSb)::T where {T <: AbstractArray{Float64}}
    k = smoother.k
 
    # if !presorted
        # x, y =   _sanitizeinput(x, y)
    # end
    Ny = length(y)
    LLSSbvals =  zeros(Float64, Ny)
    
    @inbounds @simd  for i in 1:Ny # @inbounds @simd 
        indlow = max(i - k, 1) - min(0, Ny - i - k + 1):i - 1
        indhigh = i:min(i + k, Ny) + min(0, i - k)
        xind = vcat(x[indlow], x[indhigh])
        yind = vcat(y[indlow], y[indhigh])

        fxnew = lin_reg(xind, yind)
        LLSSbvals[i] = fxnew(x[i])[1]
    end
    return  LLSSbvals
end


function do_smoothing_updating_bug(x::T, y::T, smoother::LLSSb)::T where {T <: AbstractArray{Float64}}
    k = smoother.k
    Ny = length(y)
    LLSSbvals =  zeros(Float64, Ny)
    C = 0.0
    V = 0.0
    # Calculate first point manually
    ind = 1:k + 1
    Nind = length(ind)
    xmean = mean(x[ind])
    ymean = mean(y[ind])
    @inbounds for i in ind
        tmp1 = x[i] - xmean
        C += (tmp1) * (y[i] - ymean)
        V += tmp1^2
    end
    β = C / V
    α = ymean - β * xmean
    LLSSbvals[1] = α + β * x[1]
    # Save first LLSS approximation
   # Do the next points and distringuish boundary stuff
        # Start at the 2nd point
    @inbounds @simd for i = 2:Ny
        if (i <= k + 1) # Add a "right" point
            xm = (Nind * xmean + x[k + i]) / (Nind + 1)
            ym = (Nind * ymean + y[k + i]) / (Nind + 1)
            C += (Nind + 1) / Nind * (x[k + i] - xm) * (y[k + i] - ymean)
            V += (Nind + 1) / Nind * (x[k + i] - xm)^2
            Nind = Nind + 1
            β = C / V
            α = -β * xmean + ymean
        elseif (k + 2 <= i && i <= Ny - k) # Add a "right" point, remove a "left" point
            xmean = (Nind * xmean + x[k + i]) / (Nind + 1)
            ymean = (Nind * ymean + y[k + i]) / (Nind + 1)
            C += (Nind + 1) / Nind * (x[k + i] - xmean) * (y[k + i] - ymean)
            V += (Nind + 1) / Nind * (x[k + i] - xmean)^2
            Nind = Nind + 1
            # Now the left one
            C  -= Nind / (Nind - 1) * (x[i - k - 1] - xmean) * (y[i - k - 1]  - ymean);
            V -=  Nind / (Nind - 1) * (y[i - k - 1] - xmean)^2;
            xmean = (Nind * xmean - x[i - k - 1]) / (Nind - 1);
            ymean = (Nind * ymean - y[i - k - 1]) / (Nind - 1);
            Nind = Nind - 1
            β = C / V
            α = -β * xmean + ymean
        elseif (Ny - k + 1 <= i  && i <= Ny)
            C  -= Nind / (Nind - 1) * (x[i - k - 1] - xmean) * (y[i - k - 1]  - ymean);
            V -=  Nind / (Nind - 1) * (y[i - k - 1] - xmean)^2;
            xmean = (Nind * xmean - x[i - k - 1]) / (Nind - 1);
            ymean = (Nind * ymean - y[i - k - 1]) / (Nind - 1);
   
            Nind = Nind - 1
            β = C / V
            α = -β * xmean + ymean
        else
            @warn "Did I forget anything?!" i Ny k
        end
        LLSSbvals[i] = α + β * x[i]

    end

    return  LLSSbvals
end


# Smoothing in the smoother.k*2+1 box but calculating abs(y-smoothedvals) plus do LOOCV.
function loocv(x::AbstractArray{Float64}, y::AbstractArray{Float64}, smoother::Smoother)
    k = smoother.k
    # if !presorted
        # x, y =   _sanitizeinput(x, y)
    # end
    Nx = length(x)
    Ny = length(y)
    ysmoothed = do_smoothing(x, y, smoother)
    # preallocate
    cv = zeros(Float64, (Nx,))
    @inbounds @simd for i in 1:Nx
        ind_low = max(i - k, 1) - min(0, Nx - i - k + 1):i 
        ind_high =  i + 1:min(i + k, Nx) + min(0, i - k)
        xind = vcat(x[ind_low], x[ind_high])
        xmean = mean(xind)
        denom = (1.0 - 1.0 / (2 * k) - (x[i] - xmean) / var(xind; corrected = false, mean = xmean) ) 
        cv[i] = (y[i] - ysmoothed[i] ) / denom # As in  A VARUBLE SPAN SMOOTHER by Friedman 1984
    end
    return  abs.(cv), ysmoothed
end


Base.String(k::LLSSb) = "Smoothed_LLSSb($(2 * k.k + 1))"

##


mutable struct Kernelsmooth <: Smoother
    reg::Float64
    smoothk::Kernel
end


function do_smoothing(x::T, y::T, smoother::Kernelsmooth)::T where {T <: AbstractArray{Float64}}
    # x, y =   _sanitizeinput(x, y)
    x =  Array{Float64,1}(x)
    y =  Array{Float64,1}(y)
    Nx = length(x)
    Ny = length(y)
   
    xeval =  collect(LinRange(x[1], x[end], Nx))
    kInterpolant = Kernelregression.get_kernel_interpolant(x, y, smoother.smoothk, smoother.reg)
    return kInterpolant(xeval)
end

function Base.String(ks::Kernelsmooth)
    try
        string("Smoothed_Mercer_", String(ks.smoothk), "($(ks.smoothk.σ))")
    catch
        string("Smoothed_Mercer_", String(ks.smoothk))
    end
end





mutable struct NWKernelsmooth <: Smoother
    smoothk::Kernel
end

function do_smoothing(x::T, y::T, smoother::NWKernelsmooth)::T where {T <: AbstractArray{Float64}}
    # x, y =   _sanitizeinput(x, y)
    Nx = length(x)
    Ny = length(y)
   
    m(a, b) =  Kernelregression.evalKernel(smoother.smoothk, a .- b)
    retval = zeros(size(x))
    @inbounds @simd for i in 1:Nx
        tmp = m(x[i], x)
        retval[i] = mean((tmp ./ mean(tmp)) .* y)
    end
    return    SArray{Tuple{Nx}}(retval)
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
        smoother.smoothk.σ =   0.9 * minimum(std(Y), iqr(Y) / 1.34) * N^(-1 / 5) # ((4 * std(Y)^5) / (3 * N)).^(1 / 5)
        println("We're doing it with k = $(mysmoother.k)")
    catch e
        println("We don't have a k, as we're probably using kernels.")
    end
end



mutable struct FRSS <: Smoother
    initial_Js::Array{Float64}
    medium_J::Float64
    final_J::Float64
    function FRSS(inJs, medJ, finalJ)
        new(sort(unique(inJs)), medJ, finalJ)
    end
end


@fastmath function do_smoothing(x::T, y::T, smoother::FRSS)::T where {T <: AbstractArray{Float64}}
    ## STEP 0: Prepare
    # x, y =   _sanitizeinput(x, y)
    Nx = length(x)
    Ny = length(y)

    standardsmooth = LLSSb

    ## STEP 1
    # Do the first three smooths according to initial_Js:
    # Store the three smoothed curves and get the residuals
    initial_Js_array =  repeat(smoother.initial_Js', Nx, 1) # Want column major.

    initial_smooth =   zeros(Float64, Nx, length(smoother.initial_Js))
    initial_cv_residuals =   zeros(Float64, Nx, length(smoother.initial_Js))
    @inbounds  for (i, J) in enumerate(Nx .* smoother.initial_Js)
        lin_smoother =  standardsmooth(J) 
        # local presorted = true
        initial_cv_residuals[:,i], initial_smooth[:,i] = loocv(x, y, lin_smoother)
    end


    ## STEP 2: Smooth residual curve
    initial_residuals_smoothed =  zeros(Float64, size(initial_cv_residuals))# Reuse memory 
    # Now smooth the three residuals with the medium_J
    lin_smoother =  standardsmooth(smoother.medium_J * Nx) 
    @inbounds @simd for i in 1:size(initial_cv_residuals, 2)
        initial_residuals_smoothed[:,i] = do_smoothing(x, initial_cv_residuals[:,i], lin_smoother)
    end

    ## STEP 3: Get the Js with best residuals 
    # From the three curves, take always the J with the best residual.
    best_Js = initial_Js_array[argmin(initial_residuals_smoothed, dims = 2)]

    ## STEP 4: Smooth this curve with medium smoother
    lin_smoother =  standardsmooth(smoother.medium_J * Nx) 
    smoothed_best_Js =  do_smoothing(x, best_Js, lin_smoother) # Trajectory of "best choice for J".

    # Remove Js outside the initially given values:
    clamp!(Array(smoothed_best_Js), smoother.initial_Js[1], smoother.initial_Js[end])
    # Then interpolate
    interpolated_smooth = zeros(Float64, Nx, 1)
    @inbounds @simd for i in 1:Nx
     # 1. find two curves to interpolate between. 
        j1  = 0
        # First find closest
        if abs(smoothed_best_Js[i] - smoother.initial_Js[1]) < abs(smoothed_best_Js[i] - smoother.initial_Js[2])
            j1 = 1
        else
            j1 = 2
        end
        j2 = 3
        interpolated_smooth[i] = (initial_smooth[i,j1] - initial_smooth[i,j2]) / (smoother.initial_Js[j1] - smoother.initial_Js[j2]) * (smoothed_best_Js[i] - smoother.initial_Js[j2]) + initial_smooth[i,j2]     
    end
    # Final smooth
    lin_smoother =  standardsmooth(smoother.final_J * Nx) 
    final_smooth = do_smoothing(x, interpolated_smooth, lin_smoother)
    return final_smooth
end

Base.String(frss::FRSS) = "Smoothed_FRSS" 





# @inline function _sanitizeinput(x::Array{Float64}, y::Array{Float64})
#     # x, y = sort_two_arrays_native(x, y)
#     # Nx = length(x)
#     # Ny = length(y)
#     # @assert (Nx == Ny) "Arrays must have the same length."
#     return x, y
# end

function _sanitize_k(k)
    k = k / 2
    return Int64.(round.(k, digits = 0))
end

function do_smoothing(x::AbstractArray, y::AbstractArray, smoothers::Smoother)
    for sm in smoothers::Vector{Smoother}
        y = do_smoothing(x, y, sm)
    end
    return y
end

# In case, we gave it several smoothers, do them in this order every time
function do_smoothing(x::AbstractArray, y::AbstractArray, smoothers::Array{<:Smoother})
    for sm in smoothers::Vector{Smoother}
        y = do_smoothing(x, y, sm)
    end
    return y
end