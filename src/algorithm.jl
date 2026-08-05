# The ACE backfitting algorithm.

"""
    get_sortidx(X) -> (sort_idx, sort_idx_back)

Return the permutation that sorts `X` ascending and its inverse.
"""
@stable function get_sortidx(X::AbstractVector)::Tuple{Vector{Int64},Vector{Int64}}
    N = length(X)
    sort_idx = sortperm(X)
    sort_idx_back = zeros(Int64, N)
    sort_idx_back[sort_idx] = 1:N
    return sort_idx, sort_idx_back
end

"""
    generate_bivariate_data([T], N, σ_x, σ_noise, seed)

Generate a bivariate test problem `Y = exp(X³ + ε)` with `X ~ N(0, σ_x)`.
Returns `(X, Y)` with `X` an `N × 1` matrix and `Y` an `N × 1` matrix.
"""
function generate_bivariate_data(::Type{T}=Float64, N::Int=200, σ_x::Real=1.0,
                                 σ_noise::Real=1.0, seed=nothing) where {T<:Real}
    rng = isnothing(seed) ? Random.Xoshiro() : Random.Xoshiro(seed)
    eps_err = T(σ_x) .* randn(rng, T, N)
    X = randn(rng, T, N)
    Y = exp.(X .^ 3 .+ T(σ_noise) .* eps_err)
    return reshape(X, N, 1), reshape(Y, N, 1)
end

# Backward-compatible positional form: generate_bivariate_data(N, σ_x, σ_noise, seed)
generate_bivariate_data(N::Int, args...) = generate_bivariate_data(Float64, N, args...)

"""
    ε²(Φ_x, Θ_y) -> Float64

The fraction of variance unexplained: `E[(Θ(Y) - ΣΦ(X))²] / Var(Θ(Y))`.
"""
@stable function ε²(Φ_x::AbstractArray, Θ_y::AbstractArray)::Float64
    sum_Φ = ndims(Φ_x) == 2 ? vec(sum(Φ_x; dims=2)) : vec(Φ_x)
    Θ_vec = vec(Θ_y)
    err = mean((Θ_vec .- sum_Φ) .^ 2)
    var_Θ = var(Θ_vec)
    return var_Θ > 0 ? err / var_Θ : 0.0
end

"""
    stoch_normalize!(out, X)

Standardize `X` in place into `out` (zero mean, unit variance).
"""
@stable function stoch_normalize!(out::AbstractArray, X::AbstractArray)
    n = length(X)
    μ = sum(X) / n
    σ = sqrt(sum((x - μ)^2 for x in X) / (n - 1))
    @. out = σ > 0 ? (X - μ) / σ : zero(eltype(out))
    return out
end

stoch_normalize(X::AbstractArray) = stoch_normalize!(similar(X), X)

"""
    sum_wo_theta!(out, θ, x, i)

`out[j] = θ[j] - Σ_{k≠i} x[j,k]` — the partial residual of `θ` after removing
the contribution of column `i`.
"""
@stable function sum_wo_theta!(out::AbstractArray, θ::AbstractArray, x::AbstractArray, i::Int)
    @inbounds @simd for j in 1:size(x, 1)
        out[j] = θ[j] - sum(view(x, j, 1:i-1)) - sum(view(x, j, i+1:size(x, 2)))
    end
    return out
end

# Conditional expectation via a local smoother, evaluated on the original
# (unsorted) grid using the precomputed sort indices.
function 𝔼_conditional(Y::AbstractVector, X::AbstractVector, smoother, sindx, bindx)
    X_sorted = view(X, sindx)
    Y_sorted = view(Y, sindx)
    smoothed = LocalSmoothers.do_smoothing(X_sorted, Y_sorted, smoother)
    return smoothed[bindx]
end

function 𝔼_conditional!(out::AbstractArray{T}, Y::AbstractVector, X::AbstractVector, smoother, sindx, bindx) where {T<:Real}
    X_sorted = view(X, sindx)
    Y_sorted = view(Y, sindx)
    smoothed = LocalSmoothers.do_smoothing(X_sorted, Y_sorted, smoother)
    out .= T.(smoothed[bindx])
    return out
end

"""
    ace_run(myace::ACEsim) -> ACEres

Run the ACE backfitting algorithm: alternating conditional expectations until
the unexplained variance `ε²` stops decreasing (or iteration limits are hit).
"""
@stable function ace_run(myace::ACEsim{T,S}) where {T,S<:AbstractArray}
    start = time()
    X, Y = myace.X, myace.Y
    Nx, m_parameter = size(X)

    # Preallocate arrays
    sIx = Array{Int64}(undef, Nx, m_parameter)
    bsIx = Array{Int64}(undef, Nx, m_parameter)
    for i in 1:m_parameter
        sIx[:, i], bsIx[:, i] = get_sortidx(vec(X[:, i]))
    end
    sIy, bsIy = get_sortidx(vec(Y))

    Θ_y = copy(Y)
    Φ_x = copy(X)
    θ_without_Φ_k = copy(Θ_y)

    conv_err = Vector{Float64}(undef, myace.itermax_outer * myace.itermax_inner)
    Φ_x_sum = Vector{Float64}(undef, Nx)
    temp_mean = Vector{Float64}(undef, m_parameter)

    e_old = Inf
    e_new = ε²(Φ_x, Θ_y)
    conv_err_idx = 0
    totalcount = 0

    for _ in 1:myace.itermax_outer
        for _ in 1:myace.itermax_inner
            e_old = e_new
            myace.multiloopversion == :fresh && (Φ_x .= 0.0)
            for k in 1:m_parameter
                sum_wo_theta!(θ_without_Φ_k, Θ_y, Φ_x, k)
                Φ_x[:, k] .= 𝔼_conditional(vec(θ_without_Φ_k), vec(X[:, k]), myace.smoother, sIx[:, k], bsIx[:, k])
                # center the transformation
                col_view = view(Φ_x, :, k)
                temp_mean[k] = sum(col_view) / length(col_view)
                col_view .-= temp_mean[k]
            end

            e_new = ε²(Φ_x, Θ_y)
            conv_err_idx += 1
            conv_err[conv_err_idx] = abs(e_old - e_new)
            totalcount += 1

            abs(e_old - e_new) ≤ myace.errorbound && break
        end

        # Update Θ_y: E[ΣΦ(X) | Y]
        sum!(Φ_x_sum, Φ_x)
        𝔼_conditional!(Θ_y, Φ_x_sum, vec(Y), myace.smoother, sIy, bsIy)
        stoch_normalize!(Θ_y, Θ_y)

        e_new = ε²(Φ_x, Θ_y)
        abs(e_old - e_new) ≤ myace.errorbound && break
    end

    return ACEres(
        X=X, Y=Y, Φ_x=Φ_x, Θ_y=Θ_y,
        sIx=sIx, sIy=sIy, bsIx=bsIx, bsIy=bsIy,
        conv_err=view(conv_err, 1:conv_err_idx),
        r_orig=m_parameter == 1 ? [cor(vec(X[:, 1]), vec(Y))] : [cor(vec(X[:, i]), vec(Y)) for i in 1:m_parameter],
        r²=m_parameter == 1 ? [cor(vec(Φ_x), vec(Θ_y))] : [cor(vec(Φ_x[:, i]), vec(Θ_y)) for i in 1:m_parameter],
        ρ=ε²(Φ_x, Θ_y),
        AARD=100.0 / length(X) * sum(abs.(X .- Y) ./ abs.(Y)),
        t=time() - start,
        itercount=totalcount,
        accuracy=myace.errorbound,
    )
end
