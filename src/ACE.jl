"""
Placeholder for a short summary about ACE.
"""
module ACE
# Base.Experimental.@optlevel 3
    # abstract type ACEsim end
    abstract type Smoother end
    export Smoother,run , ACEsim,generate_bivariate_data,   do_smoothing, guess_parameters!,ε², stoch_normalize
    # using PkgTemplates
    # t = Template(; user = "nildt", disable_defaults = [Git])
    using Random
    using LinearAlgebra
    using StatsPlots
    # using Revise
    using Statistics
    using BenchmarkTools
    using Test
    using Formatting:printfmt
    using Printf
    using Interpolations
    using DocStringExtensions
    using LaTeXStrings
    using StaticArrays
    using HybridArrays
    using Infiltrator
    using Parameters
    using StatsBase
    # using PyPlot
    # using ProgressMeter
    include("utils.jl")
        # Generate Test dataset which is correlated. 
        # SEED = Int64();
    include("Smoother.jl")

    # pygui(true)

@with_kw struct ACEres{S<:AbstractArray}
    X::S 
    Y::S 
    Φ_x::S
    Θ_y::S
    sIx::AbstractArray
    sIy::AbstractArray
    bsIx::AbstractArray
    bsIy::AbstractArray
    conv_err::AbstractArray
    plot_view_bounds::AbstractArray= [[(minimum(X),maximum(X)),(minimum(Y),maximum(Y))], [(minimum(X),maximum(X)),(minimum(Φ_x),maximum(Φ_x))],[(minimum(Y),maximum(Y)),(minimum(Θ_y),maximum(Θ_y))],[(minimum(Φ_x),maximum(Φ_x)),(minimum(Θ_y),maximum(Θ_y))]]
    plot_fcs::AbstractArray = []
    scale_factors::AbstractArray= [ zeros(2), zeros(2)]
    r_orig::AbstractArray
    r²::AbstractArray
    # spearman_orig::Float64
    # spearman_r²::Float64
    ρ::Float64 # max(exp(phi(y)*sum(phi_xi)))
    # RMSE::Float64
    AARD::Float64
    t::Float64 
    itercount::Int64
    description::String="ACE_simulation_result"
end

 struct ACEsim{T,S <: AbstractArray}
    "X data"
    X::S
    "Y data"
    Y::S
    "Smoothing function"
    smoother::T
    "Error bound"
    errorbound::Float64
    "Max iterations"
    itermax_inner::Int64
    "Max iterations"
    itermax_outer::Int64
end

ACEsim(X::S, Y::S, smoother::T, errorbound::Float64 = 1E-4,  itermax_inner::Int64 = 50, itermax_outer::Int64 = 500)   where  {T,S} = ACEsim{T,S}(X, Y, smoother, errorbound, itermax_inner, itermax_outer)


function get_sortidx(X::T where T <:  AbstractArray)::Tuple{Vector{Int64},Vector{Int64}}
    N = length(X)
    # We always only look at the first dimension.
    # @infiltrate
    sort_idx =  sortperm(X) # Weill can't Sortperm on SArray
    sort_idx_back = zeros(Int64, (N,))
    sort_idx_back[sort_idx] = 1:N
    return Array{Int64,1}(sort_idx[:]), Array{Int64,1}(sort_idx_back[:])
end



 function generate_bivariate_data(N = 200, σ_x = 1, σ_noise = 1, vargs...)
    rng = []
    if length(vargs) > 0
        rng = MersenneTwister(vargs[1])
    else
        rng = MersenneTwister()
    end
    eps_err = σ_x .* randn(rng, Float64, (N,))
    X = randn(rng, Float64, (N,)) # collect(LinRange(0.0, 4 * pi, N))#
    Y = exp.(X.^3 .+ σ_noise .* eps_err)
    return SVector{N,Float64}(X), SVector{N,Float64}(Y)
end


function get_uniform_distributed_data(N)
    rng = MersenneTwister()
    lb = 0.0
    ub = 5.0
    dims = (N, 1)
    return abs(ub - lb) .* (rand(rng,  Float64, dims)) .+ lb
end

  
# Generate a Nx3 dataset: Y,X1,X2
    function generate_multivariate_data(N = 200, σ_x1 = 1.0, σ_x2 = 1.0, σ_noise = 1.0, vargs...)
    rng = []
    if length(vargs) > 0
        rng = MersenneTwister(vargs[1])
    else
        rng = MersenneTwister()
    end

    eps_err = randn(rng, Float64, (N,))
    X1 = get_uniform_distributed_data(N)
    X2 = get_uniform_distributed_data(N)
    X3 = get_uniform_distributed_data(N)
    Y = X1.^2 .+ sin.(X2) .+ σ_noise .* eps_err
    return  [X1 X2 X3], Y
end

 function 𝔼_conditional(Y::Array{Float64}, X::Array{Float64},  myace::ACEsim, sindx::Vector{Int64}, bindx::Vector{Int64})
     # E(Y|X): u(x) ... (however x is implicitly given.)
    X = X[sindx]
    Y = Y[sindx]
    smoother = myace.smoother
    # @infiltrate
    sol =  do_smoothing(X, Y, smoother)[bindx]
    # Scope?
    Y = Y[bindx]
    X = X[bindx]
    return sol
end


function RMSE(Ytrue::Array{Float64}, Yestimated::Array{Float64})::Float64
    return sqrt(mean((Ytrue .- Yestimated).^2))
end


function  AARD(Y::Array{Float64}, X::AbstractMatrix)::Float64
    return 100.0 ./length(X) * sum(abs.(X.-Y)./Y)
end


function ε²(Φ_x::Array{Float64}, Θ_y::Array{Float64})::Float64
            # Please input already transformed variables
    err = mean((Θ_y .- sum(Φ_x, dims = 2)).^2) 
    err = err ./ var(Θ_y) # We don't care for scaling factors.
    # @debug "In unexp var" size(Φ_x) size(Θ_y) err1 err
    return err
end

    function MSE(X::Array{Float64}, Y::Array{Float64})::Float64
    sum((Y .- X).^2) / length(X)
end


 function stoch_normalize(X::Array{Float64})::Array{Float64}
    tmpmean = mean(X)
    return (X  .- tmpmean) ./ std(X; corrected = true, mean = tmpmean)
end

  # Multivariate ACE values
  # Restart version
# function ACEmulti_obsolete(myace::ACEsim{T,S})where {T,S <: AbstractArray} 
#     @info "Startin multivariate ACE, the predictor variables have a size of $(size(myace.X))"
#     # Normalize Mean an Variance, save the transformation
#     X = myace.X # Predictor variables
#     @show Nx, m_parameter = size(X)
#     Y = myace.Y # Response VEctor

#     # Sortindex of all predcitors
#     sIx = Array{Int64}(undef, Nx, m_parameter)
#     bsIx = Array{Int64}(undef, Nx, m_parameter)
#     for i in 1:m_parameter
#         @debug "Whatsup" Vector{Float64,1}(X[:,i])
#         sIx[:,i], bsIx[:,i] = get_sortidx(Array{Float64,1}(@view X[:,i]))
#     end
#     # SOrt response variables
#     sIy, bsIy = get_sortidx(Y)
#         # Preallocate
#     Θ_y = stoch_normalize(Y)
#     Θ_1 = Θ_y
#     Φ_1 = zeros(Float64, Nx, m_parameter)
#     Φ_x = zeros(Float64, Nx, m_parameter) # Start with zeros

#     err_candidate = 1.E6
#     err_new = 0.0

#     abserr = err_candidate
#     errorbound =    myace.errorbound 
        
#     i = 0 # Counter outer loop
#     itermax_outer = myace.itermax_outer
#     itermax_inner = myace.itermax_inner

#     itercount_inner = 0
#     itercount_outer = 0

#     @inbounds while abserr > errorbound &&  i < itermax_outer #  || i < 5
#         j = 0 # Counter inner loop
#         while abserr > errorbound &&  j < itermax_inner
#             j = j + 1 # Count immediately (as indexing starts at 1)
#             @inbounds @simd for k in 1:m_parameter 
#                 Φ_1[:,k] = 𝔼_conditional(X, Θ_y,  myace,  sIx, bsIx) # E_y(...)
#                 Φ_x[:,k] =   stoch_normalize(Φ_1[:,k]) #  Φ_1  .- mean(Φ_1) # normalize mean #   stoch_normalize(Φ_1)# stoch_normalize(Φ_1)
#             end
#             err_new = ε²(Φ_x, Θ_y)
#             abserr = abs(err_new - err_candidate) 
#             itercount_inner = itercount_inner + 1
#         end
#         i += 1 # Count immediately (as indexing starts at 1)

#         err_candidate = err_new

#         Θ_1 =  𝔼_conditional(Y, Φ_x, myace, sIy, bsIy) # E_x(Phi(x)|Y)
#         Θ_y =  stoch_normalize(Θ_1) # Θ_1 .- mean(Θ_1) #
#         # Θ_y  = Θ_1
#         err_new = ε²(Φ_x, Θ_y)
#         abserr = abs(err_new - err_candidate) 
#                 # println("In iter $i we get an error of $  abserr to the loop before.")
#                 # printfmt("In Iteration $i we get an error of {:.9f}",abs(err_new - err_candidate))
#         # @info "ACE Loop:" itercount_outer
#         itercount_outer = itercount_outer + 1
#     end
#     # println("Did $i iterations!")
#             # p = ACE_4_plot(X, Y, Φ_x, Θ_y)
#             # display(p)
#             # @info "Dbg" Θ_y Φ_x
  

#     var_unexp = ε²(Φ_x, Θ_y)
#     thisemse = MSE(Φ_x, Θ_y)    
#     correl = cor(Φ_x, Θ_y)
#     itercount_total = itercount_outer * itercount_inner
#     return  Φ_x, Θ_y, itercount_inner, itercount_outer, itercount_total, var_unexp, thisemse, correl
# end

# function ACE_bivariate_basic(myace::ACEsim{T,S})where {T,S <: AbstractArray} 
#     Nx = length(myace.X)
#     X = myace.X
#     Y = myace.Y
#     sIx, bsIx = get_sortidx(vec(X))
#     sIy, bsIy = get_sortidx(vec(Y))
#     Θ_y = stoch_normalize(Y)
#     Θ_candidate = copy(Θ_y)
#     Θ_1 = copy(Θ_y)
#     Φ_1 = stoch_normalize(X)
#     Φ_x = stoch_normalize(X)
#     @debug "What is Φₓ?" typeof(Φ_x)
#     Φ_candidate = copy(Φ_x)
#         # @debug "Check sort X:" issorted(Φ_1) issorted(Φ_x) issorted(Φ_candidate) 
#         # @debug "Check sort Y:" issorted(Θ_1) issorted(Θ_y) issorted(Θ_candidate) 
#     @debug "Before the loop." ε²(Θ_candidate, Φ_candidate)  ε²(Θ_y, Φ_x)

#     i = 0
#     #  h1 = Plots.scatter(Φ_x,Θ_y)
#     conv_err = []
#     @inbounds while ((abs(ε²(Θ_y, Φ_x) - ε²(Θ_candidate, Φ_candidate)) > myace.errorbound) || i < 2 ) && i < myace.itermax_outer
#         push!(conv_err, ε²(Θ_y, Φ_x))
#         Φ_candidate = Φ_x
#         Θ_candidate = Θ_y
#         Φ_x = 𝔼_conditional(Θ_y, X, myace, sIx, bsIx)
#         Θ_1 = 𝔼_conditional(Φ_x, Y, myace, sIy, bsIy) 
#         Θ_y = stoch_normalize(Θ_1)
#         @debug "Iteration" i  ε²(Θ_candidate, Φ_candidate) - ε²(Θ_y, Φ_x)
#         #   if mod(i,2)==0 
#             # Plots.scatter!(h1,Φ_candidate,Θ_candidate[bsIy],s=1,alpha=1/(i+1))
#         #   end
#         i += 1
#     end
#     # PyPlot.scatter(collect(1:length(conv_err)),log.(conv_err))
#     # gcf()
#     Φ_x = Φ_candidate  # (candidate was successful)
#     Θ_y = Θ_candidate

# #   var_unexp = ε²(Φ_x, Θ_y)
# #     thisemse = MSE(Φ_x, Θ_y)
# #     correl = cor(Φ_x, Θ_y)
#     # itercount_total = itercount_outer * itercount_inner
#     return  Φ_x, Θ_y, sIx, sIy, conv_err
# end

@inline function sum_without(X::AbstractArray, k::Int64)
    @fastmath return sum(hcat(X[:,1:k - 1], X[:,k + 1:end]), dims = 2)
end

function run(myace::ACEsim{T,S}) where {T,S <: AbstractArray} 
    start = time()

    X = myace.X # Predictor variables
    Nx, m_parameter = size(X)
    Y = myace.Y # Response VEctor
    # Sortindex of all predcitors
    sIx = Array{Int64}(undef, Nx, m_parameter)
    bsIx = Array{Int64}(undef, Nx, m_parameter)
    for i in 1:m_parameter
        sIx[:,i], bsIx[:,i] = get_sortidx(X[:,i])
    end
    # SOrt response variables
    sIy, bsIy = get_sortidx(vec(Y))
    Θ_y = stoch_normalize(copy(Y))
    Θ_candidate = copy(Θ_y)

    Φ_x =  copy(X)# zeros(Float64, Nx, m_parameter) # zeros(Float64, Nx, m_parameter) # Start with zeros
    for i in 1:m_parameter
        Φ_x[:,i] .=   Φ_x[:,i] .- mean(Φ_x[:,i])
    end
    Φ_candidate = copy(Φ_x)

    itermax_outer = myace.itermax_outer
    itermax_inner = myace.itermax_inner
    e_old = Inf64
    e_new =  ε²(Θ_y, Φ_x)
    errorbound =    myace.errorbound 


    conv_err = []
    totalcount = 0
    i = 1
    while (abs(e_old - e_new) > myace.errorbound || i < 2)  && i <= myace.itermax_outer
        j = 1
        while (abs(e_old - e_new) > myace.errorbound || j < 2) && j <= myace.itermax_inner
            Φ_x = Φ_candidate 
            e_old = e_new
            @inbounds for k in 1:m_parameter
                Φ_candidate[:,k] = 𝔼_conditional(Θ_y .- sum_without(Φ_candidate, k), X,  myace,  sIx[:,k], bsIx[:,k]) # E_y(...)
                Φ_candidate[:,k] = Φ_candidate[:,k] .- mean(Φ_candidate[:,k])
            end
            e_new =  ε²(Θ_y, Φ_candidate)
            @debug "Inner loop" (e_old - e_new )  i j
            push!(conv_err, abs(e_old - e_new))
            j += 1
            totalcount  += 1
        end
        e_old = e_new;
        Θ_y .=  Θ_candidate
        Θ_candidate  = 𝔼_conditional(sum(Φ_x, dims = 2), Y, myace, sIy, bsIy) 
        Θ_candidate  = stoch_normalize(Θ_candidate)
        i += 1
        e_new =  ε²(Θ_candidate, Φ_x)
        @debug "outer loop" (e_old - e_new ) i j
        push!(conv_err, abs(e_old - e_new))
    end

    r_orig =cor(X, Y)
    r² = cor(Φ_x, Θ_y)
    # spearman_orig = StatsBase.corspearman(vec(X), vec(Y))
    # spearman_r² = StatsBase.corspearman(vec(Φ_x), vec(Θ_y))
    ρ =     ε²(Φ_x, Θ_y)
    # mRMSE = RMSE(Φ_x, Θ_y)
    mAARD = AARD(Φ_x, Θ_y)
    t =  time() - start
    itercount= totalcount
return  ACEres(X=X,Y=Y, Φ_x= Φ_x, Θ_y=Θ_y, sIx=sIx, sIy=sIy, bsIx=bsIx,bsIy= bsIy, conv_err=conv_err, r_orig =  r_orig , r² =  r², ρ  =  ρ , AARD =  mAARD, t  =  t,itercount=itercount)
    # return res
end




# Plot recipie.
@recipe function f(bf::ACEres;transform=false, full=true,dpi=500,plotsize=1. .*(2200,2200))
#       if length(bf.X) == 0  || !(typeof(bf.X) <: AbstractVector) ||
#         !(typeof(bf.Φ_x) <: AbstractVector)
#         error("Benchmark has wrong dimensions, or ACE solution wasn't set yet.  Got: $(typeof(bf))")
#     end

    markershape --> :circle
    markersize  --> 2
           # # set up the subplots
            link --> :none
            size-->plotsize
  xguide --> "x"
    yguide --> "y"
                    margin -->20Plots.px

        if full
            X = bf.X
            Y = bf.Y
            Φ_x = bf.Φ_x
            Θ_y = bf.Θ_y

            # remove nan:

            X[isnan.(X)] .= -Inf
            Y[isnan.(Y)] .= -Inf
            Φ_x[isnan.(Φ_x)] .= -Inf
            Θ_y[isnan.(Θ_y)] .= -Inf


            plot_view_bounds = bf.plot_view_bounds
            plot_fcs = bf.plot_fcs

            n = size(X,2)
     

            # framestyle := [:shared :shared :shared :shared]
            grid := false
            layout :=  @layout [a{0.05h}; StatsPlots.grid(3,  n)  b{0.3w}; c{0.1h}] # ;b{0.2h}
            seriestype := :scatter
                background_color := RGB(0.2, 0.2, 0.2)
                dpi:= dpi
                 bottom_margin:=50Plots.px
                left_margin:=100Plots.px
                colorbar:= false
                legend:= false
   
                markerstrokewidth := 0
                e2 = string(round(abs((ε²(Φ_x[bf.sIx], Θ_y[bf.sIy])));digits = 4))
               @show bf
                 mytit = join(["\nACE result:\n","ε² = $e2 "])
                title:= mytit
           
              @series begin
                    seriestype := :scatter
                    framestyle:=:none
                        subplot := 1
                end

 # do the first n:
                for i in 1:Int64(n)
                    @series begin
                        seriestype := :scatter
                        title:=""
                        xguide := "X$i"
                            yguide := "Y"
                            subplot := i+1
                                    xlims := plot_view_bounds[1][1]
                                    ylims:= plot_view_bounds[1][2]
                                X[:,i],Y
                        end
            end
            #second row 
                       for i in 1:Int64(n)
                    @series begin
                            seriestype := :scatter
                    title:=""
                         xlabel --> "X"
                ylabel --> "Φ(X$i)"
                    # xlims := plot_view_bounds[2][1]
                            ylims:= plot_view_bounds[2][2]
                    subplot := n+1+i
                        X[:,i],Φ_x[:,i]
                end
            end

            # Last row
            for i in 1:Int64(n)
                           @series begin
                    title:=""
                xlabel --> "Φ(X$i)"
                ylabel --> L"\Theta(Y)"
                                        # xlims := plot_view_bounds[4][1]
                            ylims:= plot_view_bounds[4][2]
                    subplot := 2*n+1+i
                             Φ_x[:,i], Θ_y

                end
            end


                           @series begin
                    title:=""
                xlabel --> "Y"
                ylabel --> L"\Theta(Y)"
                           xlims := plot_view_bounds[3][1]
                            ylims:= plot_view_bounds[3][2]
                    subplot := 3*n+1+1
                Y,Θ_y

                end


                                @series begin
                                        seriestype := :scatter
                                            markersize  := 3
                    title:=""
                xlabel --> "Iterations"
                ylabel --> "ε"
                subplot := 3*n+1+1+1
                 Array{Float64}(collect( 1:length(bf.conv_err))),   Array{Float64}(bf.conv_err)
                end

                       
        else 

        if transform && length(bf.Φ_x ) != 0 
            x:= bf.Φ_x
            y:= bf.Θ_y

        else
        x:=bf.X
        y:=bf.Y
        end
    end
    # ()
end




end            