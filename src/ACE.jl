module ACE

include("/Users/nilswildt/ResilioSync/ac125867/03_projects/17_ACE_for_MCMC/ACE.jl/src/Kernelregression/Kernelregression.jl")
using .Kernelregression

include("/Users/nilswildt/ResilioSync/ac125867/03_projects/17_ACE_for_MCMC/ACE.jl/src/Smoother.jl")
using .Smoothers

export run, ACEsim, generate_bivariate_data, do_smoothing, ε², stoch_normalize, ACEres

using Parameters
using CairoMakie
using Printf
using LinearAlgebra
using StatsBase 
using DispatchDoctor
using ConcreteStructs
using LaTeXStrings
using Random
using Statistics

include("Kernelregression/Kernelregression.jl")
using .Kernelregression
include("ACEBenchmarksMakie.jl")
using .ACEBenchmarksMakie
using CairoMakie

bf = ACEBenchmarksMakie.f_b1(2_000, 1, "normal", 1.0, 0.3, true, 42, false)


Base.@kwdef @concrete struct ACEres{T<:Real}
    X::AbstractArray
    Y::AbstractArray
    Φ_x::AbstractArray{T}
    Θ_y::AbstractArray{T}
    sIx::AbstractArray{Int64}
    sIy::AbstractArray{Int64}
    bsIx::AbstractArray{Int64}
    bsIy::AbstractArray{Int64}
    conv_err::AbstractArray{T}
    plot_view_bounds::AbstractArray = [
        [(minimum(X), maximum(X)), (minimum(Y), maximum(Y))],
        [(minimum(X), maximum(X)), (minimum(Φ_x), maximum(Φ_x))],
        [(minimum(Y), maximum(Y)), (minimum(Θ_y), maximum(Θ_y))],
        [(minimum(Φ_x), maximum(Φ_x)), (minimum(Θ_y), maximum(Θ_y))]
    ]
    plot_fcs::Vector = []
    scale_factors::Vector = [zeros(2), zeros(2)]
    r_orig::AbstractArray{T}
    r²::AbstractArray{T}
    ρ::T
    AARD::T
    t::Float64
    itercount::Int64
    accuracy::T
    description::String = "ACE_simulation_result"
end

Base.@kwdef @concrete struct ACEsim{T,S1<:AbstractArray,S2<:AbstractArray}
    X::S1
    Y::S2
    smoother::T
    errorbound::Float64
    itermax_inner::Int64
    itermax_outer::Int64
    multiloopversion::Symbol
end

@stable function ACEsim(X::S1, Y::S2, smoother::T;
                errorbound::Float64=1e-4, 
                itermax_inner::Int64=50, 
                itermax_outer::Int64=500, 
                multiloopversion=:fresh) where {T,S1<:AbstractArray,S2<:AbstractArray}
    ACEsim{T,S1,S2}(X, Y, smoother, errorbound, itermax_inner, itermax_outer, multiloopversion)
end

@stable function get_sortidx(X::AbstractArray)::Tuple{Vector{Int64},Vector{Int64}}
    N = length(X)
    sort_idx = sortperm(X)
    sort_idx_back = zeros(Int64, N)
    sort_idx_back[sort_idx] = 1:N
    return sort_idx, sort_idx_back
end

@stable function generate_bivariate_data(::Type{T}=Float64, N=200, σ_x=0.1, σ_noise=1, seed=nothing) where T<:Real
    rng = isnothing(seed) ? Random.Xoshiro() : Random.Xoshiro(seed)
    eps_err = T(σ_x) .* randn(rng, T, N)
    X = randn(rng, T, N)
    Y = exp.(X .^ 3 ).+ T(σ_noise) .* eps_err
    return reshape(X, N, 1), vec(Y)  # Return X as Nx1 matrix and Y as vector
end

# For backward compatibility
generate_bivariate_data(N::Int, args...) = generate_bivariate_data(Float64, N, args...)

@stable function 𝔼_conditional!(out::AbstractArray{T}, Y::AbstractArray, X::AbstractArray, 
                                myace::ACEsim, sindx, bindx) where T<:Real
    X_vec = reshape(X, length(X))
    Y_vec = reshape(Y, length(Y))
    X_sorted = @view X_vec[sindx]
    Y_sorted = @view Y_vec[sindx]
    outm = Smoothers.do_smoothing(X_sorted, Y_sorted, myace.smoother)
    out .= T.(outm[bindx])
end

@stable function 𝔼_conditional!(out::AbstractArray{T}, Y::AbstractArray, X::AbstractArray, 
                                myace::ACEsim{Vector{<:Smoother}}, sindx, bindx) where T<:Real
    X_vec = reshape(X, length(X))
    Y_vec = reshape(Y, length(Y))
    X_sorted = @view X_vec[sindx]
    Y_sorted = @view Y_vec[sindx]
    outm = Smoothers.do_smoothing(X_sorted, Y_sorted, myace.smoother)
    out .= T.(outm[bindx])
end

@stable function 𝔼_conditional!(out::AbstractArray{T}, Y::AbstractArray, X::AbstractArray, 
                                myace::ACEsim{Vector{S}}, sindx, bindx) where {T<:Real, S<:Smoother}
    X_vec = reshape(X, length(X))
    Y_vec = reshape(Y, length(Y))
    X_sorted = @view X_vec[sindx]
    Y_sorted = @view Y_vec[sindx]
    outm = Smoothers.do_smoothing(X_sorted, Y_sorted, myace.smoother)
    out .= T.(outm[bindx])
end

@stable function 𝔼_conditional(Y::AbstractArray, X::AbstractArray, 
                               myace::ACEsim, sindx, bindx)
    X_vec = reshape(X, length(X))
    Y_vec = reshape(Y, length(Y))
    X_sorted = @view X_vec[sindx]
    Y_sorted = @view Y_vec[sindx]
    smoothed = Smoothers.do_smoothing(X_sorted, Y_sorted, myace.smoother)
    return smoothed[bindx]
end

@stable function 𝔼_conditional(Y::AbstractArray, X::AbstractArray, 
                               myace::ACEsim{Vector{<:Smoother}}, sindx, bindx)
    X_vec = reshape(X, length(X))
    Y_vec = reshape(Y, length(Y))
    X_sorted = @view X_vec[sindx]
    Y_sorted = @view Y_vec[sindx]
    smoothed = Smoothers.do_smoothing(X_sorted, Y_sorted, myace.smoother)
    return smoothed[bindx]
end

@stable function 𝔼_conditional(Y::AbstractArray, X::AbstractArray, 
                               myace::ACEsim{Vector{S}}, sindx, bindx) where S<:Smoother
    X_vec = reshape(X, length(X))
    Y_vec = reshape(Y, length(Y))
    X_sorted = @view X_vec[sindx]
    Y_sorted = @view Y_vec[sindx]
    smoothed = Smoothers.do_smoothing(X_sorted, Y_sorted, myace.smoother)
    return smoothed[bindx]
end

# Handle mixed smoother arrays with explicit type annotation
@stable function 𝔼_conditional(Y::AbstractArray, X::AbstractArray, 
                               myace::ACEsim{Vector{<:Smoother}}, sindx, bindx)
    X_vec = reshape(X, length(X))
    Y_vec = reshape(Y, length(Y))
    X_sorted = @view X_vec[sindx]
    Y_sorted = @view Y_vec[sindx]
    smoothed = Smoothers.do_smoothing(X_sorted, Y_sorted, myace.smoother)
    return smoothed[bindx]
end

@stable function ε²(Φ_x::AbstractArray, Θ_y::AbstractArray)
    T = promote_type(eltype(Φ_x), eltype(Θ_y))
    # Calculate the sum of Φ_x across columns (for multivariate case)
    if ndims(Φ_x) == 2
        sum_Φ = vec(sum(Φ_x, dims=2))
    else
        sum_Φ = vec(Φ_x)
    end
    
    Θ_vec = vec(Θ_y)
    err = mean((Θ_vec .- sum_Φ) .^ 2)
    
    var_Θ = var(Θ_vec)
    return var_Θ > 0 ? T(err / var_Θ) : zero(T)
end

@stable function stoch_normalize!(out::AbstractArray, X::AbstractArray)
    T = promote_type(eltype(out), eltype(X))
    n = length(X)
    μ = sum(X) / n
    σ = sqrt(sum((x - μ)^2 for x in X) / (n - 1))
    if σ > 0
        @. out = T((X - μ) / σ)
    else
        out .= zero(T)
    end
    return out
end

@stable function stoch_normalize(X::AbstractArray)
    T = eltype(X)
    out = similar(X)
    stoch_normalize!(out, X)
    return out
end

@stable function sum_wo_theta!(out::AbstractArray, θ::AbstractArray, 
                               x::AbstractArray, i::Int)
    @simd for j in 1:size(x,1)
        out[j] = θ[j] - sum(view(x, j, 1:i-1)) - sum(view(x, j, i+1:size(x,2)))
    end
end

@stable function run(myace::ACEsim{TS,S1,S2}) where {TS,S1<:AbstractArray,S2<:AbstractArray}
    T = promote_type(eltype(S1), eltype(S2))
    start = time()
    X, Y = myace.X, myace.Y
    Nx, m_parameter = size(X)
    
    # Preallocate arrays
    sIx = Array{Int64}(undef, Nx, m_parameter)
    bsIx = Array{Int64}(undef, Nx, m_parameter)
    for i in 1:m_parameter
        sIx[:,i], bsIx[:,i] = get_sortidx(X[:,i])
    end
    sIy, bsIy = get_sortidx(vec(Y))
    
    Θ_y = T.(copy(vec(Y)))
    Φ_x = T.(copy(X))
    θ_without_Φ_k = similar(Θ_y)
    
    # Preallocations
    conv_err = Vector{T}(undef, myace.itermax_outer * myace.itermax_inner)
    Φ_x_sum = Vector{T}(undef, Nx)
    temp_mean = Vector{T}(undef, m_parameter)
    
    e_old = T(Inf)
    e_new = ε²(Φ_x, Θ_y)
    conv_err_idx = 0
    totalcount = 0
    
    for i in 1:myace.itermax_outer
        for j in 1:myace.itermax_inner
            e_old = e_new
            myace.multiloopversion == :fresh && (Φ_x .= zero(T))
            
            for k in 1:m_parameter
                sum_wo_theta!(θ_without_Φ_k, Θ_y, Φ_x, k)
                smoothed = 𝔼_conditional(θ_without_Φ_k, X[:,k], myace, sIx[:,k], bsIx[:,k])
                Φ_x[:,k] .= T.(smoothed)
                # Center the transformation
                col_view = view(Φ_x, :, k)
                temp_mean[k] = mean(col_view)
                col_view .-= temp_mean[k]
            end
            
            e_new = ε²(Φ_x, Θ_y)
            conv_err_idx += 1
            conv_err[conv_err_idx] = abs(e_old - e_new)
            totalcount += 1
            
            abs(e_old - e_new) ≤ myace.errorbound && break
        end
        
        # Update Θ_y: E[sum(Φ_x) | Y]
        sum!(Φ_x_sum, Φ_x)  # Sum across columns
        𝔼_conditional!(Θ_y, Φ_x_sum, Y, myace, sIy, bsIy)
        stoch_normalize!(Θ_y, Θ_y)
        
        e_new = ε²(Φ_x, Θ_y)
        abs(e_old - e_new) ≤ myace.errorbound && break
    end
    
    # Calculate correlations
    r_orig = if m_parameter == 1
        T[cor(vec(X), vec(Y))]
    else
        T[cor(X[:,i], vec(Y)) for i in 1:m_parameter]
    end
    
    r² = if m_parameter == 1
        T[cor(vec(Φ_x), Θ_y)]
    else
        T[cor(Φ_x[:,i], Θ_y) for i in 1:m_parameter]
    end
    
    return ACEres{T}(
        X=X, Y=Y, Φ_x=Φ_x, Θ_y=Θ_y,
        sIx=sIx, sIy=sIy, bsIx=bsIx, bsIy=bsIy,
        conv_err=view(conv_err, 1:conv_err_idx),
        r_orig=r_orig, r²=r²,
        ρ=ε²(Φ_x, Θ_y),
        AARD=T(100.0/length(Y) * sum(abs.(vec(Y) .- mean(Y)) ./ abs.(vec(Y)))),
        t=time() - start,
        itercount=totalcount,
        accuracy=T(myace.errorbound)
    )
end

# Plotting function using CairoMakie
function plot_ace_results(bf::ACEres; transform=false, full=true, 
                          figsize=(800, 600))
    if full
        X = bf.X
        Y = bf.Y
        Φ_x = bf.Φ_x
        Θ_y = bf.Θ_y
        n = size(X, 2)
        
        fig = Figure(resolution=figsize)
        
        # Title
        e2 = round(abs(ε²(Φ_x, Θ_y)); digits=4)
        Label(fig[0, :], text=L"ACE result: $\varepsilon^2 = %$e2$", 
        fontsize=24, font="bold")
        
        # Create grid of subplots
        # First row: X vs Y
        for i in 1:n
            ax = Axis(fig[1, i], xlabel=L"X_{%$i}", ylabel=L"Y")
            scatter!(ax, X[:, i], vec(Y), markersize=4, color=:blue)
        end
        
        # Second row: X vs Φ(X)
        for i in 1:n
            ax = Axis(fig[2, i], xlabel=L"X_{%$i}", ylabel=L"\Phi(X_{%$i})")
            scatter!(ax, X[:, i], Φ_x[:, i], markersize=4, color=:green)
        end
        
        # Third row: Φ(X) vs Θ(Y)
        for i in 1:n
            ax = Axis(fig[3, i], xlabel=L"\Phi(X_{%$i})", ylabel=L"\Theta(Y)")
            scatter!(ax, Φ_x[:, i], Θ_y, markersize=4, color=:red)
        end
        
        # Y vs Θ(Y) plot
        ax = Axis(fig[1:2, n+1], xlabel=L"Y", ylabel=L"\Theta(Y)")
        scatter!(ax, vec(Y), Θ_y, markersize=4, color=:purple)
        
        # Convergence plot
        ax = Axis(fig[3, n+1], xlabel="Iterations", ylabel=L"\varepsilon",
                  yscale=log10)
        scatter!(ax, 1:length(bf.conv_err), bf.conv_err, markersize=3, color=:black)
    display(fig)
        
        return fig
    else
        fig = Figure(;size=(800, 600))
        ax = Axis(fig[1, 1])
        
        if transform && length(bf.Φ_x) != 0
            scatter!(ax, vec(bf.Φ_x), bf.Θ_y, markersize=4)
        else
            scatter!(ax, vec(bf.X), vec(bf.Y), markersize=4)
        end
    display(fig)
        
        return fig
    end
end

end # module