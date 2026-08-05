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

# One backfitting sweep over all predictors:
# φₖ ← E[θ − Σ_{j≠k} φⱼ | xₖ], each centered to zero mean. Shared by the ACE
# and AVAS response loops, which differ only in how θ(Y) is updated.
function backfit_sweep!(Φ_x, Θ_y, θ_without_Φ_k, temp_mean, X, transforms, sIx, bsIx, multiloopversion)
    multiloopversion == :fresh && (Φ_x .= 0.0)
    for k in 1:size(X, 2)
        sum_wo_theta!(θ_without_Φ_k, Θ_y, Φ_x, k)
        Φ_x[:, k] .= transform_fit(transforms[k], vec(X[:, k]), vec(θ_without_Φ_k), sIx[:, k], bsIx[:, k])
        col_view = view(Φ_x, :, k)
        temp_mean[k] = sum(col_view) / length(col_view)
        col_view .-= temp_mean[k]
    end
    return Φ_x
end

"""
    ace_run(myace::ACEsim) -> ACEres

Run ACE from the legacy [`ACEsim`](@ref) description. Prefer [`ace`](@ref),
which takes the data and options directly and supports per-variable transforms.
"""
function ace_run(myace::ACEsim)
    return ace(myace.X, myace.Y; smoother=myace.smoother,
               errorbound=myace.errorbound, itermax_inner=myace.itermax_inner,
               itermax_outer=myace.itermax_outer, multiloopversion=myace.multiloopversion)
end
