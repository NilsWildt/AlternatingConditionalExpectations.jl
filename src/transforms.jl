# Per-variable transform strategies for the backfitting steps.
#
# Each `VarTransform` estimates the transform of a single variable from its
# values and the current target (the partial residual for a predictor, or the
# response for the y-step). This is the dispatch point that lets a variable be
# smoothed, constrained monotone, linear, categorical, or periodic — mirroring
# acepack's per-column type codes (`mon`/`lin`/`cat`/`circ`).

abstract type VarTransform end

"""
    Smooth(smoother)

Estimate the transform by smoothing the target against the (sorted) variable —
the standard ACE nonparametric step. `smoother` is any `LocalSmoothers`
smoother (or a vector of them, applied as a chain).
"""
struct Smooth{S} <: VarTransform
    smoother::S
end

"""
    Monotone(smoother)

Like [`Smooth`](@ref), but the smoothed transform is projected onto monotone
functions by isotonic regression (pool-adjacent-violators), choosing whichever
of the increasing/decreasing projections better fits the smooth.
"""
struct Monotone{S} <: VarTransform
    smoother::S
end

"""
    LinearFit()

Constrain the transform to a linear (ordinary least squares) function of the
variable.
"""
struct LinearFit <: VarTransform end

"""
    Categorical()

Treat the variable as categorical: map each distinct level to the mean of the
target over that level.
"""
struct Categorical <: VarTransform end

"""
    Periodic(smoother, period)

Smooth the target against the variable on a circle of the given `period`
(wrap-around), so the transform matches at the two ends of one period.
"""
struct Periodic{S} <: VarTransform
    smoother::S
    period::Float64
end

# --- isotonic regression -------------------------------------------------

"""
    pava!(y)

In-place least-squares projection of `y` (in domain order) onto the cone of
monotone-nondecreasing sequences, by pool-adjacent-violators. `O(n)`.
"""
function pava!(y::AbstractVector{<:Real})
    Base.require_one_based_indexing(y)
    n = length(y)
    n <= 1 && return y
    val = collect(float.(y))            # block values
    wgt = ones(Int, n)                  # block weights
    len = ones(Int, n)                  # block lengths (points pooled)
    nb = 0
    for i in 1:n
        nb += 1
        val[nb] = y[i]; wgt[nb] = 1; len[nb] = 1
        while nb > 1 && val[nb-1] > val[nb]
            w = wgt[nb-1] + wgt[nb]
            val[nb-1] = (val[nb-1] * wgt[nb-1] + val[nb] * wgt[nb]) / w
            wgt[nb-1] = w
            len[nb-1] += len[nb]
            nb -= 1
        end
    end
    pos = 1
    for b in 1:nb
        for _ in 1:len[b]
            y[pos] = val[b]
            pos += 1
        end
    end
    return y
end

# Project `y` onto monotone functions, returning the better of the increasing
# and decreasing least-squares fits (ACE transforms are defined up to sign, so
# the direction is chosen from the data).
function isotonic(y::AbstractVector{<:Real})
    inc = pava!(collect(float.(y)))
    dec = pava!(.-float.(y)); dec .= .-dec
    sse(p) = sum((y[i] - p[i])^2 for i in eachindex(y, p))
    return sse(inc) <= sse(dec) ? inc : dec
end

# --- transform fitting ---------------------------------------------------

# Fit the transform of one variable. `x` is the variable's values, `target` the
# quantity being explained; `sidx`/`bidx` are the ascending sort permutation of
# `x` and its inverse (precomputed once per fit). Returns the fitted transform
# aligned to `x` (original, unsorted order).
function transform_fit(t::Smooth, x, target, sidx, bidx)
    return 𝔼_conditional(target, x, t.smoother, sidx, bidx)
end

function transform_fit(t::Monotone, x, target, sidx, bidx)
    φ = 𝔼_conditional(target, x, t.smoother, sidx, bidx)
    return isotonic(φ[sidx])[bidx]      # monotone in ascending-x order, then unsort
end

function transform_fit(::LinearFit, x, target, sidx, bidx)
    n = length(x)
    x̄ = sum(x) / n
    ȳ = sum(target) / n
    sxx = sum((xi - x̄)^2 for xi in x)
    sxy = sum((x[i] - x̄) * (target[i] - ȳ) for i in eachindex(x, target))
    a = sxx > 0 ? sxy / sxx : zero(sxy)
    b = ȳ - a * x̄
    return @. a * x + b
end

function transform_fit(::Categorical, x, target, sidx, bidx)
    n = length(x)
    xs = x[sidx]                        # equal levels are contiguous once sorted
    ts = target[sidx]
    φs = similar(ts, float(eltype(ts)))
    i = 1
    while i <= n
        j = i
        while j < n && xs[j+1] == xs[i]
            j += 1
        end
        level_mean = sum(@view ts[i:j]) / (j - i + 1)
        φs[i:j] .= level_mean
        i = j + 1
    end
    return φs[bidx]
end

function transform_fit(t::Periodic, x, target, sidx, bidx)
    n = length(x)
    P = t.period
    # Replicate the data one period below and above so the smoother sees a
    # wrap-around neighbourhood; keep the central copy.
    xe = vcat(x .- P, x, x .+ P)
    te = vcat(target, target, target)
    pe = sortperm(xe)
    se = LocalSmoothers.do_smoothing(xe[pe], te[pe], t.smoother)
    φ_ext = se[invperm(pe)]
    return φ_ext[n+1:2n]
end
