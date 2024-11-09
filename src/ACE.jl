module ACE

# Include Smoothers as a submodule
include("Smoother.jl")  # This should export the Smoothers module

# Make sure to use ACE.Smoothers namespace when referring to smoother types
using .Smoothers  # This makes Smoothers types available within ACE

export run, ACEsim, generate_bivariate_data, do_smoothing,  ε², stoch_normalize, ACEres
using Parameters
using Plots
# using Random, LinearAlgebra, Statistics, Parameters
# using StaticArrays, StatsBase
# using DocStringExtensions
# using StatsPlots # for plotting recipe
# using Interpolations
using Printf
using LinearAlgebra
using StatsBase 
using DispatchDoctor
using ConcreteStructs



include("Kernelregression/Kernelregression.jl")
using .Kernelregression

Base.@kwdef @concrete struct ACEres
    X::AbstractArray
    Y::AbstractArray
    Φ_x::AbstractArray
    Θ_y::AbstractArray
    sIx::AbstractArray
    sIy::AbstractArray
    bsIx::AbstractArray
    bsIy::AbstractArray
    conv_err::AbstractArray
    plot_view_bounds::AbstractArray = [
        [(minimum(X), maximum(X)), (minimum(Y), maximum(Y))],
        [(minimum(X), maximum(X)), (minimum(Φ_x), maximum(Φ_x))],
        [(minimum(Y), maximum(Y)), (minimum(Θ_y), maximum(Θ_y))],
        [(minimum(Φ_x), maximum(Φ_x)), (minimum(Θ_y), maximum(Θ_y))]
    ]
    plot_fcs::Vector = []
    scale_factors::Vector = [zeros(2), zeros(2)]
    r_orig::AbstractArray
    r²::AbstractArray
    ρ::Float64
    AARD::Float64
    t::Float64
    itercount::Int64
    accuracy::Float64
    description::String = "ACE_simulation_result"
end

Base.@kwdef @concrete struct ACEsim{T,S<:AbstractArray}
    X::S
    Y::S
    smoother::T
    errorbound::Float64
    itermax_inner::Int64
    itermax_outer::Int64
    multiloopversion::Symbol
end

@stable function ACEsim(X::S, Y::S, smoother::T;
                errorbound::Float64=1e-4, 
                itermax_inner::Int64=50, 
                itermax_outer::Int64=500, 
                multiloopversion=:fresh) where {T,S}
    ACEsim{T,S}(X, Y, smoother, errorbound, itermax_inner, itermax_outer, multiloopversion)
end

@stable function get_sortidx(X::AbstractArray)::Tuple{Vector{Int64},Vector{Int64}}
    N = length(X)
    sort_idx = sortperm(X)
    sort_idx_back = zeros(Int64, N)
    sort_idx_back[sort_idx] = 1:N
    return sort_idx, sort_idx_back
end

@stable function generate_bivariate_data(N=200, σ_x=1, σ_noise=1, seed=nothing)
    rng = isnothing(seed) ? Xoshiro() : Xoshiro(seed)
    eps_err = σ_x .* randn(rng, Float64, N)
    X = randn(rng, Float64, N)
    Y = exp.(X .^ 3 .+ σ_noise .* eps_err)
    return SVector{N,Float64}(X), SVector{N,Float64}(Y)
end

@stable function 𝔼_conditional!(out, Y, X, myace::ACEsim, sindx, bindx)
    X_vec = reshape(X,:) #vec(X)
    Y_vec = reshape(Y,:) #vec(Y)
    X_sorted = @view X_vec[sindx]
    Y_sorted = @view Y_vec[sindx]
    outm = Smoothers.do_smoothing(X_sorted, Y_sorted, myace.smoother)
    out .= outm[bindx]
end

@stable function ε²(Φ_x::AbstractArray, Θ_y::AbstractArray)::Float64
    err = StatsBase.mean((Θ_y .- sum(Φ_x, dims=2)) .^ 2)
    return err / var(Θ_y)
end

@stable function stoch_normalize!(out::AbstractArray, X::AbstractArray)
    n = length(X)
    μ = sum(X) / n
    σ = sqrt(sum((x - μ)^2 for x in X) / (n - 1))
    @. out = (X - μ) / σ
    return out
end

@stable function stoch_normalize(X::AbstractArray)
    out = similar(X)
    stoch_normalize!(out, X)
    return out
end

@stable function sum_wo_theta!(out, θ, x, i::Int)
    @simd for j in 1:size(x,1)
        out[j] = θ[j] - sum(view(x, j, 1:i-1)) - sum(view(x, j, i+1:size(x,2)))
    end
end


@stable function run(myace::ACEsim{T,S}) where {T,S<:AbstractArray}
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
    
    Θ_y = copy(Y)
    Φ_x = copy(X)
    θ_without_Φ_k = copy(Θ_y)
    
    # New preallocations
    conv_err = Vector{Float64}(undef, myace.itermax_outer * myace.itermax_inner)
    Φ_x_col = Vector{Float64}(undef, Nx)  # for column operations
    Φ_x_sum = Vector{Float64}(undef, Nx)  # for sum storage
    temp_mean = Vector{Float64}(undef, m_parameter)  # for mean calculations
    
    e_old = Inf
    e_new = ε²(Θ_y, Φ_x)
    conv_err_idx = 0
    totalcount = 0
    
    for i in 1:myace.itermax_outer
        for j in 1:myace.itermax_inner
            e_old = e_new
            myace.multiloopversion == :fresh && (Φ_x .= 0.0)
            
            for k in 1:m_parameter
                sum_wo_theta!(θ_without_Φ_k, Θ_y, Φ_x, k)
                𝔼_conditional!(view(Φ_x, :, k), θ_without_Φ_k, X[:,k], myace, sIx[:,k], bsIx[:,k])
                
                # Use preallocated array for mean calculation
                col_view = view(Φ_x, :, k)
                temp_mean[k] = sum(col_view) / length(col_view)
                col_view .-= temp_mean[k]
            end
            
            e_new = ε²(Θ_y, Φ_x)
            conv_err_idx += 1
            conv_err[conv_err_idx] = abs(e_old - e_new)
            totalcount += 1
            
            abs(e_old - e_new) ≤ myace.errorbound && break
        end
        
        # Use preallocated sum array
        sum!(Φ_x_sum, Φ_x)
        𝔼_conditional!(Θ_y, Φ_x_sum, Y, myace, sIy, bsIy)
        stoch_normalize!(Θ_y, Θ_y)
        
        e_new = ε²(Θ_y, Φ_x)
        abs(e_old - e_new) ≤ myace.errorbound && break
    end
    
    return ACEres(
        X=X, Y=Y, Φ_x=Φ_x, Θ_y=Θ_y,
        sIx=sIx, sIy=sIy, bsIx=bsIx, bsIy=bsIy,
        conv_err=view(conv_err, 1:conv_err_idx),
        r_orig=cor(X, Y), r²=cor(Φ_x, Θ_y),
        ρ=ε²(Φ_x, Θ_y),
        AARD=100.0/length(X) * sum(abs.(X .- Y) ./ abs.(Y)),
        t=time() - start,
        itercount=totalcount,
        accuracy=myace.errorbound
    )
end

# Plot recipie.
@recipe function f(bf::ACEres; transform=false, full=true, dpi=500, plotsize=2.0 .* (1200, 800))
    #       if length(bf.X) == 0  | | !(typeof(bf.X) <: AbstractVector) ||
    #         !(typeof(bf.Φ_x) <: AbstractVector)
    #         error("Benchmark has wrong dimensions, or ACE solution wasn't set yet.  Got: $(typeof(bf))")
    #     end

    markershape --> :circle
    markersize --> 2
    # # set up the subplots
    link --> :none
    size --> plotsize
    xguide --> "x"
    yguide --> "y"
    margin --> 20Plots.px

    if full
        X = bf.X
        Y = bf.Y
        Φ_x = bf.Φ_x
        Θ_y = bf.Θ_y

        # # remove nan:

        # X[isnan.(X)] .= 0.0 #  -Inf64
        # Y[isnan.(Y)] .= 0.0 #  -Inf64
        # Φ_x[isnan.(Φ_x)] .= 0.0 #  -Inf64
        # Θ_y[isnan.(Θ_y)] .= 0.0 #  -Inf64

        #! Add some catching NaNs!


        plot_view_bounds = bf.plot_view_bounds
        plot_fcs = bf.plot_fcs

        n = size(X, 2)


        # framestyle := [:shared :shared :shared :shared]
        grid := false
        layout := @layout [a{0.05h}; grid(3, n) b{0.3w}; c{0.1h}] # ;b{0.2h}
        seriestype := :scatter
        background_color := RGB(0.2, 0.2, 0.2)
        dpi := dpi
        colorbar := false
        legend := false

        markerstrokewidth := 0
        e2 = string(round(abs((ε²(Φ_x[bf.sIx], Θ_y[bf.sIy]))); digits=4))
        @show bf
        mytit = join(["\nACE result:\n", "ε² = $e2 "])
        title := mytit

        # Header (subplot 1)
        @series begin
            seriestype := :scatter
            framestyle := :none
            subplot := 1
        end

        # Main grid (subplots 2 to 3n+1)
        for i in 1:Int64(n)
            @series begin
                seriestype := :scatter
                title := ""
                xguide := "X$i"
                yguide := "Y"
                subplot := i + 1
                # xlims := plot_view_bounds[1][1]
                # ylims:= plot_view_bounds[1][2]
                X[:, i], Y
            end
        end
        #second row 
        for i in 1:Int64(n)
            @series begin
                seriestype := :scatter
                title := ""
                xlabel --> "X"
                ylabel --> "Φ(X$i)"
                # xlims := plot_view_bounds[2][1]
                # ylims:= plot_view_bounds[2][2]
                subplot := n + 1 + i
                X[:, i], Φ_x[:, i]
            end
        end

        # Last row
        for i in 1:Int64(n)
            @series begin
                title := ""
                xlabel --> "Φ(X$i)"
                ylabel --> "Θ(Y)"
                # xlims := plot_view_bounds[4][1]
                # ylims:= plot_view_bounds[4][2]
                subplot := 2 * n + 1 + i
                Φ_x[:, i], Θ_y

            end
        end


        @series begin
            title := ""
            xlabel --> "Y"
            ylabel --> "Θ(Y)"
            #    xlims := plot_view_bounds[3][1]
            # ylims:= plot_view_bounds[3][2]
            subplot := 3 * n + 1 + 1
            Y, Θ_y

        end


        @series begin
            seriestype := :scatter
            markersize := 3
            title := ""
            xlabel --> "Iterations"
            ylabel --> "ε"
            subplot := 3 * n + 1 + 1 + 1
            Array{Float64}(collect(1:length(bf.conv_err))), Array{Float64}(bf.conv_err)
        end


    else

        if transform && length(bf.Φ_x) != 0
            x := bf.Φ_x
            y := bf.Θ_y

        else
            x := bf.X
            y := bf.Y
        end
    end
    # ()
end




end