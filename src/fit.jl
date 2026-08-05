# The public fitting interface: `ace` / `avas` / `predict`.
#
# Both algorithms share one backfitting engine (`_fit`); they differ only in the
# response update — ACE takes a conditional expectation (a `VarTransform` on Y),
# AVAS applies the variance-stabilizing transform. Per-predictor transforms let
# individual variables be smoothed, monotone, linear, categorical, or periodic.

"""
    FitControls(; errorbound, itermax_inner, itermax_outer, multiloopversion)

Convergence controls for [`ace`](@ref) / [`avas`](@ref): the ε² tolerance, the
inner/outer iteration caps, and the backfitting reset policy (`:fresh` zeroes
the predictor transforms at the start of each inner loop).
"""
Base.@kwdef struct FitControls
    errorbound::Float64 = 1e-4
    itermax_inner::Int64 = 50
    itermax_outer::Int64 = 500
    multiloopversion::Symbol = :fresh
end

_asmatrix(A::AbstractVector) = reshape(A, length(A), 1)
_asmatrix(A::AbstractMatrix) = A

# Expand a transform specification to one entry per predictor.
_expand_transforms(t::VarTransform, m) = [t for _ in 1:m]
function _expand_transforms(ts::AbstractVector, m)
    length(ts) == m ||
        throw(ArgumentError("need one xtransform per predictor: got $(length(ts)) for $m columns"))
    return ts
end

# Unified backfitting engine. `method` is :ace or :avas; `var_smoother` is used
# only by the AVAS variance step.
function _fit(X, Y, xtransforms::AbstractVector, ytransform, method::Symbol,
              ctrl::FitControls, var_smoother)
    start = time()
    Nx, m = size(X)

    sIx = Array{Int64}(undef, Nx, m)
    bsIx = Array{Int64}(undef, Nx, m)
    for i in 1:m
        sIx[:, i], bsIx[:, i] = get_sortidx(vec(X[:, i]))
    end
    sIy, bsIy = get_sortidx(vec(Y))

    Θ_y = copy(Y)
    method === :avas && stoch_normalize!(Θ_y, Θ_y)
    Φ_x = copy(X)
    θ_without_Φ_k = copy(Θ_y)

    conv_err = Vector{Float64}(undef, ctrl.itermax_outer * ctrl.itermax_inner)
    Φ_x_sum = Vector{Float64}(undef, Nx)
    temp_mean = Vector{Float64}(undef, m)

    e_old = Inf
    e_new = ε²(Φ_x, Θ_y)
    conv_err_idx = 0
    totalcount = 0
    converged = false

    for _ in 1:ctrl.itermax_outer
        for _ in 1:ctrl.itermax_inner
            e_old = e_new
            backfit_sweep!(Φ_x, Θ_y, θ_without_Φ_k, temp_mean, X,
                           xtransforms, sIx, bsIx, ctrl.multiloopversion)
            e_new = ε²(Φ_x, Θ_y)
            conv_err_idx += 1
            conv_err[conv_err_idx] = abs(e_old - e_new)
            totalcount += 1
            abs(e_old - e_new) ≤ ctrl.errorbound && break
        end

        if method === :avas
            variance_stabilize!(Θ_y, Φ_x, Φ_x_sum, var_smoother)
        else
            sum!(Φ_x_sum, Φ_x)
            Θ_y .= reshape(transform_fit(ytransform, vec(Y), Φ_x_sum, sIy, bsIy), size(Θ_y))
            stoch_normalize!(Θ_y, Θ_y)
        end

        e_new = ε²(Φ_x, Θ_y)
        if abs(e_old - e_new) ≤ ctrl.errorbound
            converged = true
            break
        end
    end

    sum!(Φ_x_sum, Φ_x)
    return ACEres(
        X=X, Y=Y, Φ_x=Φ_x, Θ_y=Θ_y,
        sIx=sIx, sIy=sIy, bsIx=bsIx, bsIy=bsIy,
        conv_err=view(conv_err, 1:conv_err_idx),
        r_orig=m == 1 ? [cor(vec(X[:, 1]), vec(Y))] : [cor(vec(X[:, i]), vec(Y)) for i in 1:m],
        r²=m == 1 ? [cor(vec(Φ_x), vec(Θ_y))] : [cor(vec(Φ_x[:, i]), vec(Θ_y)) for i in 1:m],
        ρ=ε²(Φ_x, Θ_y),
        AARD=AARD(vec(Θ_y), Φ_x_sum),
        t=time() - start,
        itercount=totalcount,
        accuracy=ctrl.errorbound,
        converged=converged,
    )
end

function _resolve_smoother(smoother, xtransforms, ytransform)
    smoother === nothing &&
        throw(ArgumentError("pass `smoother`, or supply `xtransforms` (and `ytransform` for ace)"))
    return smoother
end

"""
    ace(X, Y; smoother, xtransforms, ytransform, kwargs...) -> ACEres

Fit ACE (Breiman & Friedman, 1985): alternating conditional expectations of the
predictor transforms and the response transform.

`X` is an `N×m` matrix (or length-`N` vector for a single predictor) and `Y` a
length-`N` vector or `N×1` matrix. `smoother` is any `LocalSmoothers` smoother
and provides the default [`Smooth`](@ref) transform for every variable; pass
`xtransforms` (one [`VarTransform`](@ref) per predictor, or a single one applied
to all) and/or `ytransform` to constrain individual transforms
(`Monotone`, `LinearFit`, `Categorical`, `Periodic`). Remaining keywords are
[`FitControls`](@ref).

Access results as `model.tx` (φ), `model.ty` (θ), `model.rsq`, `model.iters`,
`model.converged`; use [`predict`](@ref) for new data.
"""
function ace(X, Y; smoother=nothing, xtransforms=nothing, ytransform=nothing,
             errorbound=1e-4, itermax_inner=50, itermax_outer=500, multiloopversion=:fresh)
    Xm, Ym = _asmatrix(X), _asmatrix(Y)
    m = size(Xm, 2)
    xt = xtransforms === nothing ? [Smooth(_resolve_smoother(smoother, xtransforms, ytransform)) for _ in 1:m] :
         _expand_transforms(xtransforms, m)
    yt = ytransform === nothing ? Smooth(_resolve_smoother(smoother, xtransforms, ytransform)) : ytransform
    ctrl = FitControls(errorbound, itermax_inner, itermax_outer, multiloopversion)
    return _fit(Xm, Ym, xt, yt, :ace, ctrl, smoother)
end

"""
    avas(X, Y; smoother, xtransforms, kwargs...) -> ACEres

Fit AVAS (Tibshirani, 1988): like [`ace`](@ref) for the predictor transforms,
but the response transform is updated by a variance-stabilizing transform (see
[`ctsub`](@ref)) rather than a conditional expectation, which yields a monotone
response transform. `smoother` drives both the backfitting and the variance
step. `xtransforms` may constrain individual predictors as in [`ace`](@ref).
"""
function avas(X, Y; smoother=nothing, xtransforms=nothing,
              errorbound=1e-4, itermax_inner=50, itermax_outer=500, multiloopversion=:fresh)
    Xm, Ym = _asmatrix(X), _asmatrix(Y)
    m = size(Xm, 2)
    sm = _resolve_smoother(smoother, xtransforms, nothing)
    xt = xtransforms === nothing ? [Smooth(sm) for _ in 1:m] : _expand_transforms(xtransforms, m)
    ctrl = FitControls(errorbound, itermax_inner, itermax_outer, multiloopversion)
    return _fit(Xm, Ym, xt, Smooth(sm), :avas, ctrl, sm)
end

# Linear interpolation over a sorted grid, flat outside the range.
function _interp(xs::AbstractVector, ys::AbstractVector, x)
    x <= xs[1] && return float(ys[1])
    x >= xs[end] && return float(ys[end])
    j = searchsortedfirst(xs, x)
    x1, x2 = xs[j-1], xs[j]
    x2 == x1 && return float(ys[j])
    return ys[j-1] + (x - x1) / (x2 - x1) * (ys[j] - ys[j-1])
end

"""
    predict(model::ACEres, Xnew) -> Vector

Predict the transformed-response fit `Σⱼ φⱼ(xⱼ)` for new predictor rows, by
linearly interpolating each learned predictor transform over its training grid.
Returns values on the transformed-response (θ) scale.
"""
function predict(model::ACEres, Xnew::AbstractMatrix)
    Xtr, Φ = model.X, model.Φ_x
    n, m = size(Xnew)
    size(Xtr, 2) == m ||
        throw(DimensionMismatch("model has $(size(Xtr, 2)) predictors, Xnew has $m"))
    out = zeros(float(eltype(Φ)), n)
    for k in 1:m
        p = sortperm(vec(view(Xtr, :, k)))
        xs = vec(view(Xtr, :, k))[p]
        φs = vec(view(Φ, :, k))[p]
        for i in 1:n
            out[i] += _interp(xs, φs, Xnew[i, k])
        end
    end
    return out
end
predict(model::ACEres, xnew::AbstractVector) = predict(model, reshape(xnew, length(xnew), 1))
