# Prediction error metrics (ported from the original ACE.jl).

"""
    MAE(Ytrue, Yestimated)

Mean absolute error.
"""
function MAE(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    return mean(abs.(Ytrue .- Yestimated))
end

"""
    nMAE(Ytrue, Yestimated)

Mean absolute error normalized by the mean of the truth.
"""
function nMAE(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    return MAE(Ytrue, Yestimated) ./ mean(Ytrue)
end

"""
    RMSE(Ytrue, Yestimated)

Root mean squared error.
"""
function RMSE(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    return sqrt(mean((Ytrue .- Yestimated) .^ 2))
end

"""
    UFV(Ytrue, Yestimated)

Unexplained fraction of variance.
"""
function UFV(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    return mean((Ytrue .- Yestimated) .^ 2) ./ var(Ytrue)
end

"""
    myErr(Ytrue, Yestimated)

Mean of the square root of absolute errors.
"""
function myErr(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    return mean(sqrt.(abs.(Ytrue .- Yestimated) .^ 2))
end

"""
    pErr(Ytrue, Yestimated)

Mean squared error.
"""
function pErr(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    return mean((Ytrue .- Yestimated) .^ 2)
end

"""
    AARD(Ytrue, Yestimated)

Average absolute relative deviation (in percent).
"""
function AARD(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    acc = 0.0
    n = 0
        acc += abs(Yestimated[i] - yt) / abs(yt)
    end
    return n == 0 ? NaN : 100.0 * acc / n
end
