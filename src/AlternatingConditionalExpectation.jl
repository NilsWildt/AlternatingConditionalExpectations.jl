"""
Placeholder for a short summary about AlternatingConditionalExpectation.
"""
module AlternatingConditionalExpectation
    abstract type ACE end
    abstract type Smoother end
    export Smoother, ACE, Acerun,generate_bivariate_data, ACE_bivariate, ACE_4_plot, do_smoothing, guess_parameters!,unexplained_variance, stoch_normalize, ACE_multivariate
    # using PkgTemplates
    # t = Template(; user = "nildt", disable_defaults = [Git])
    using Random
    using LinearAlgebra
    using StaticArrays
    using StatsPlots
    # using Revise
    using Statistics
    using BenchmarkTools
    using Test
    using Formatting: printfmt
    using Printf
    using Interpolations
    using DocStringExtensions
    using LaTeXStrings
    using StaticArrays
    using HybridArrays
 
    # using ProgressMeter
    include("utils.jl")
        # Generate Test dataset which is correlated. 
        # SEED = Int64();
    include("Smoother.jl")
struct Acerun
    "X data"
    X::SMatrix
    "Y data"
    Y::SArray
    "Smoothing function"
    smoother::Smoother
    "Error bound"
    errorbound::Float64
    "Max iterations"
    itermax_inner::Int64
    "Max iterations"
    itermax_outer::Int64
end

Acerun(X::SMatrix, Y::SArray, smoother::Smoother, errorbound::Float64 = 1E-4,  itermax_inner::Int64 = 10, itermax_outer::Int64 = 100)  = Acerun(X, Y, smoother, errorbound, itermax_inner, itermax_outer)

function get_sortidx(X::StaticVector)
    N = length(X)
    sort_idx =  sortperm(X)
    sort_idx_back = zeros(Int64, N)
    sort_idx_back[sort_idx] = 1:N
    return sort_idx, sort_idx_back
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


    function get_uni(N)
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
    X1 =get_uni(N)
    X2 = get_uni(N)
    X3 = get_uni(N)
    Y = X1.^2 .+ sin.(X2).+ σ_noise .* eps_err
    return SVector{N,Float64}(Y), SArray{(N,3),Float64}([X1 X2 X3])
end

  @inline  function cond_exp(X::StaticVector, Y::StaticVector,  myace::Acerun, sindx::AbstractArray, bindx::AbstractArray) 
    X = X[sindx]
    Y = Y[sindx]
    smoother = myace.smoother
    return SVector{length(X),Float64}(do_smoothing(X, Y, smoother)[bindx])
end

 @inline function unexplained_variance(Φ_x::StaticVector, Θ_y::StaticVector)::Float64
            # Please input already transformed variables
    err = mean((Θ_y .- Φ_x).^2) 
    err = err ./ var(Θ_y) # We don't care for scaling factors.
    return err
end

    @inline function MSE(X::StaticVector, Y::StaticVector)::Float64
    sum((Y .- X).^2) / length(X)
end


    @inline function stoch_normalize(X::StaticVector)::StaticVector
    tmpmean = mean(X)
    return (X  .- tmpmean) ./ std(X; corrected = true, mean = tmpmean)
end

  # Multivariate ACE values
  # Restart version
function ACE_multivariate(myace::Acerun)
    @info "Startin multivariate ACE, the predictor variables have a size of $(size(myace.X))"
    # Normalize Mean an Variance, save the transformation
    X = myace.X # Predictor variables
    Nx,m_parameter = size(X)
    Y = myace.Y # Response VEctor

    # Sortindex of all predcitors
sIx = Array{Int64}(undef, Nx,3)
bsIx = Array{Int64}(undef, Nx,3)
    for i in 1:m_parameter
        sIx[:,i], bsIx[:,i] = get_sortidx(X[:,i])
    end
    # SOrt response variables
    sIy, bsIy = get_sortidx(Y)


        # Preallocate
    Θ_y = stoch_normalize(Y)
    Θ_1 = Θ_y
    Φ_1 = @SMatrix zeros(Float64,Nx,3)
    Φ_x = @SMatrix zeros(Float64,Nx,3) # Start with zeros

    err_old = Inf64
    err_new = unexplained_variance(X, Y)

    abserr = err_old
    errorbound =    myace.errorbound 
        
    i = 0 # Counter outer loop
    itermax_outer = myace.itermax_outer
    itermax_inner = myace.itermax_inner

    itercount_inner = 0
    itercount_outer = 0

    @inbounds while abserr > errorbound &&  i < itermax_outer #  || i < 5
        j = 0 # Counter inner loop
        while abserr > errorbound &&  j < itermax_inner
            j = j + 1 # Count immediately (as indexing starts at 1)
            err_old = err_new
            @inbounds @simd for k in 1:m_parameter 
                Φ_1[:,k] = cond_exp(X, Θ_y,  myace,  sIx, bsIx) # E_y(...)
                Φ_x[:,k] =   stoch_normalize(Φ_1[:,k]) #  Φ_1  .- mean(Φ_1) # normalize mean #   stoch_normalize(Φ_1)# stoch_normalize(Φ_1)
            end
            err_new = unexplained_variance(Φ_x, Θ_y)
            abserr = abs(err_new - err_old) 
            itercount_inner = itercount_inner + 1
        end
        i += 1 # Count immediately (as indexing starts at 1)

        err_old = err_new

        Θ_1 =  cond_exp(Y, Φ_x, myace, sIy, bsIy) # E_x(Phi(x)|Y)
        Θ_y =  stoch_normalize(Θ_1) # Θ_1 .- mean(Θ_1) #
        # Θ_y  = Θ_1
        err_new = unexplained_variance(Φ_x, Θ_y)
        abserr = abs(err_new - err_old) 
                # println("In iter $i we get an error of $abserr to the loop before.")
                # printfmt("In Iteration $i we get an error of {:.9f}",abs(err_new - err_old))
        # @info "ACE Loop:" itercount_outer
        itercount_outer = itercount_outer + 1
    end
    # println("Did $i iterations!")
            # p = ACE_4_plot(X, Y, Φ_x, Θ_y)
            # display(p)
            # @info "Dbg" Θ_y Φ_x
  

    var_unexp = unexplained_variance(Φ_x, Θ_y)
    thisemse = MSE(Φ_x, Θ_y)
    correl = cor(Φ_x, Θ_y)
    itercount_total = itercount_outer * itercount_inner
    return  Φ_x, Θ_y, itercount_inner, itercount_outer, itercount_total, var_unexp, thisemse, correl
end


function ACE_bivariate(myace::Acerun)
    # Normalize Mean an Variance, save the transformation
    Nx = length(X)
    X = SArray{Nx,Float64}(myace.X)
    Y = myace.Y

    sIx, bsIx = get_sortidx(X)
    sIy, bsIy = get_sortidx(Y)
        # Preallocate
    Θ_y = stoch_normalize(Y)
    Θ_1 = Θ_y
    Φ_1 = MVector{Nx,Float64}(zeros(Float64, Nx))
    Φ_x = MVector{Nx,Float64}(zeros(Float64, Nx))

    err_old = Inf64
    err_new = unexplained_variance(X, Y)

    abserr = err_old
    errorbound =    myace.errorbound 
        
    i = 0
    itermax_outer = myace.itermax_outer
    itermax_inner = myace.itermax_inner

    itercount_inner = 0
    itercount_outer = 0

    @inbounds while abserr > errorbound &&  i < itermax_outer #  || i < 5
        j = 0
        while abserr > errorbound &&  j < itermax_inner 
            err_old = err_new
            Φ_1 = cond_exp(X, Θ_y,  myace,  sIx, bsIx) # E_y(...)
            Φ_x =   stoch_normalize(Φ_1) #  Φ_1  .- mean(Φ_1) # normalize mean #   stoch_normalize(Φ_1)# stoch_normalize(Φ_1)

            err_new = unexplained_variance(Φ_x, Θ_y)
            abserr = abs(err_new - err_old) 
            j = j + 1
            itercount_inner = itercount_inner + 1
        end
        err_old = err_new

        Θ_1 =  cond_exp(Y, Φ_x, myace, sIy, bsIy) # E_x(Phi(x)|Y)
        Θ_y =  stoch_normalize(Θ_1) # Θ_1 .- mean(Θ_1) #
        # Θ_y  = Θ_1
        err_new = unexplained_variance(Φ_x, Θ_y)
        abserr = abs(err_new - err_old) 
                # println("In iter $i we get an error of $abserr to the loop before.")
                # printfmt("In Iteration $i we get an error of {:.9f}",abs(err_new - err_old))
        i += 1
        # @info "ACE Loop:" itercount_outer
        itercount_outer = itercount_outer + 1
    end
    # println("Did $i iterations!")
            # p = ACE_4_plot(X, Y, Φ_x, Θ_y)
            # display(p)
            # @info "Dbg" Θ_y Φ_x
  

    var_unexp = unexplained_variance(Φ_x, Θ_y)
    thisemse = MSE(Φ_x, Θ_y)
    correl = cor(Φ_x, Θ_y)
    itercount_total = itercount_outer * itercount_inner
    return  Φ_x, Θ_y, itercount_inner, itercount_outer, itercount_total, var_unexp, thisemse, correl
end

        
 @noinline    function ACE_4_plot(X, Y, Φ_x, Θ_y,  vargs...)
    alldpi = 300
    if length(vargs) >= 1
        alldpi = vargs[2]
    end
    p1 = StatsPlots.scatter(
                X,
                Y,
                xlabel = "X",
                ylabel = "Y",
                m = ([:circle], 2),
                bg = RGB(0.2, 0.2, 0.2),
                dpi = alldpi,
                cbar = false,
                leg = false,
                xlim = (-2.5, 2.5),
                ylim = (0, 80),
                markerstrokewidth = 0
            )
    p2 = StatsPlots.scatter(
                X,Φ_x,
                xlabel = "X",
                ylabel = L"\Phi(X)",
                m = ([:circle], 2),
                bg = RGB(0.2, 0.2, 0.2),
                dpi = alldpi,
                cbar = false,
                leg = false,
                xlim = (-2.5, 2.5),
                ylim = (-2.5, 2.5),
                markerstrokewidth = 0
            )
    p2 = plot!(p2, sort(X), sort(X).^2)
    plot!(p2, sort(X), sort(X))
    # X, Y = sort_two_arrays(X, Y)
    # p3Y, p3Θ_y = sort_two_arrays(Y, Θ_y)
    p3 = StatsPlots.scatter(
                Y,Θ_y,
                xlabel = "Y",
                ylabel = L"\Theta(Y)",
                m = ([:circle], 2),
                bg = RGB(0.2, 0.2, 0.2),
                dpi = alldpi,
                cbar = false,
                leg = false,
                xlim = (0, 80),
                ylim = (-2, 2),
                markerstrokewidth = 0
            )
    p3 =  plot!(p3, sort(Y), log.(sort(Y)))
    # plot!(p3, sort(Y), (sort(Y).^(1 / 3)))

    plot!(p3, sort(Y), abs.(log.(sort(Y))).^(1 / 3))
    # X, Φ_x
    p4 = StatsPlots.scatter(
                Φ_x, Θ_y,
                xlabel = L"\Phi(X)",
                ylabel = L"\Theta(Y)",
                m = ([:circle], 2),
                bg = RGB(0.2, 0.2, 0.2),
                dpi = alldpi,
                cbar = false,
                leg = false,
                xlim = (-3, 3),
                ylim = (-2.2, 2.5),
                markerstrokewidth = 0
            )
        # scatter!(X.^3, log.(Y), m = (:dot, 1))
    if length(vargs) >= 1
        e2 = string(round(abs((unexplained_variance(Φ_x, Θ_y)));digits = 2))
        mytit = join([vargs[1] , "1- e^2 = $e2 "])
        l = @layout [a{0.03h}; grid(2, 2)]
        title = plot(title = mytit, grid = false, showaxis = false, bottom_margin = -50Plots.px) 
        plot(title, p1, p2, p3, p4, layout = l,  size = 0.8 .* (1.6 * 800, 800))
    else
        l = @layout [a b; c d]
        plot(p1, p2, p3, p4, layout = l)
    end
end


end


