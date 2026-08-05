# AVAS: Additivity and VAriance Stabilization (Tibshirani, 1988).
#
# AVAS shares ACE's predictor backfitting exactly; it differs only in the
# response update. Where ACE sets θ(Y) = E[ΣΦ(X) | Y], AVAS replaces θ with a
# variance-stabilizing transform: it estimates the residual variance as a
# function of the fitted value and integrates its inverse square root. Because
# that integral is of a strictly positive weight, the resulting θ is monotone,
# which is what lets AVAS reproduce model transformations that ACE cannot.
#
# Reference implementation: acepack's `favas.f90` / `ctsub.f90`
# (https://github.com/vubiostat/acepack).

"""
    ctsub(u, v, y) -> ty

Cumulative trapezoidal integral of the grid function `(u, v)` (with `u` sorted
ascending) evaluated at each point of `y`:

    ty[i] = ∫_{u[1]}^{y[i]} v(s) ds

using linear interpolation of `v` between grid points and the end slope for
`y[i]` outside `[u[1], u[end]]`. Port of acepack's `ctsub`, the core of the
AVAS variance-stabilizing transform.
"""
function ctsub(u::AbstractVector{<:Real}, v::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(u) == length(v) ||
        throw(DimensionMismatch("u and v must match: $(length(u)) vs $(length(v))"))
    # The algorithm reads the grid positionally (u[j-1], u[j]); u/v are the
    # freshly sorted fitted values, so one-based indexing is the honest contract.
    Base.require_one_based_indexing(u, v)
    n = length(u)
    ty = similar(y, float(eltype(y)))
    for i in eachindex(y, ty)
        yi = y[i]
        if yi <= u[1]
            ty[i] = (yi - u[1]) * v[1]
            continue
        end
        acc = zero(eltype(ty))
        j = n + 1
        for jj in 1:n
            if yi <= u[jj]
                j = jj
                break
            end
            jj > 1 && (acc += (u[jj] - u[jj-1]) * (v[jj] + v[jj-1]) / 2)
        end
        if yi <= u[n]                       # interior: partial trapezoid to yi
            du = u[j] - u[j-1]
            slope = du == 0 ? zero(acc) : (v[j] - v[j-1]) / du
            dy = yi - u[j-1]
            acc += dy * v[j-1] + dy^2 * slope / 2
        else                                # beyond the grid: extend end slope
            acc += (yi - u[n]) * v[n]
        end
        ty[i] = acc
    end
    return ty
end

# AVAS response update: replace θ(Y) with its variance-stabilizing transform.
# Smooths log-squared residuals against the fitted value to estimate the
# log-variance, then integrates exp(-logvar/2) and restandardizes.
function variance_stabilize!(Θ_y::AbstractArray, Φ_x::AbstractArray,
                             Φ_x_sum::AbstractVector, smoother)
    sum!(Φ_x_sum, Φ_x)                      # μ = ΣⱼΦⱼ, the fitted values
    θ = vec(Θ_y)
    T = float(eltype(θ))
    δ = sqrt(eps(T))
    logr2 = @. 2 * log(abs(θ - Φ_x_sum) + δ) # log squared residuals, zero-guarded
    p = sortperm(Φ_x_sum)
    u = Φ_x_sum[p]                           # fitted values, sorted ascending
    logvar = LocalSmoothers.do_smoothing(u, logr2[p], smoother)
    v = @. exp(-logvar / 2)                  # 1/√variance at the grid points
    new_θ = ctsub(u, v, θ)
    stoch_normalize!(Θ_y, new_θ)             # recenter + unit variance
    return Θ_y
end

"""
    avas_run(myace::ACEsim) -> ACEres

Run AVAS (Tibshirani, 1988): the same alternating backfitting of the predictor
transforms as [`ace_run`](@ref), but with the response transform updated by a
variance-stabilizing transform instead of a conditional expectation. Uses the
smoother carried by `myace` for both the backfitting and the variance step.
"""
function avas_run(myace::ACEsim)
    return avas(myace.X, myace.Y; smoother=myace.smoother,
                errorbound=myace.errorbound, itermax_inner=myace.itermax_inner,
                itermax_outer=myace.itermax_outer, multiloopversion=myace.multiloopversion)
end
