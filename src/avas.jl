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
@stable function avas_run(myace::ACEsim{T,S}) where {T,S<:AbstractArray}
    start = time()
    X, Y = myace.X, myace.Y
    Nx, m_parameter = size(X)

    sIx = Array{Int64}(undef, Nx, m_parameter)
    bsIx = Array{Int64}(undef, Nx, m_parameter)
    for i in 1:m_parameter
        sIx[:, i], bsIx[:, i] = get_sortidx(vec(X[:, i]))
    end
    sIy, bsIy = get_sortidx(vec(Y))

    Θ_y = copy(Y)
    stoch_normalize!(Θ_y, Θ_y)              # AVAS starts from the standardized response
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
            backfit_sweep!(Φ_x, Θ_y, θ_without_Φ_k, temp_mean, X,
                           myace.smoother, sIx, bsIx, myace.multiloopversion)

            e_new = ε²(Φ_x, Θ_y)
            conv_err_idx += 1
            conv_err[conv_err_idx] = abs(e_old - e_new)
            totalcount += 1

            abs(e_old - e_new) ≤ myace.errorbound && break
        end

        # Update Θ_y by the variance-stabilizing transform (this is the only
        # step that differs from ace_run).
        variance_stabilize!(Θ_y, Φ_x, Φ_x_sum, myace.smoother)

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
