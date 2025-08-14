function MAE(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    return mean(abs.(Ytrue .- Yestimated))
end

function nMAE(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    return MAE(Ytrue, Yestimated) ./ mean(Ytrue)
end

function RMSE(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    return sqrt(mean((Ytrue .- Yestimated).^2))
end

function UFV(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    return mean((Ytrue .- Yestimated).^2) ./ var(Ytrue)
end

function myErr(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    return mean(sqrt.(abs.(Ytrue .- Yestimated).^2))
end

function pErr(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    return mean(((Ytrue .- Yestimated).^2))
end

function  AARD(Ytrue::AbstractArray, Yestimated::AbstractArray)::Float64
    return 100.0 ./length(Ytrue) * sum(abs.(Yestimated-Ytrue)./Ytrue)
end