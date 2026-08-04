# Result and simulation structs.

"""
    ACEres

Container returned by [`run`](@ref): the fitted transformations `Φ_x`, `Θ_y`,
the sort indices, convergence history and diagnostics.

# Fields
- `X`, `Y`: input data (as passed to [`ACEsim`](@ref))
- `Φ_x`, `Θ_y`: fitted optimal transformations
- `sIx`, `sIy`, `bsIx`, `bsIy`: sort indices used during fitting
- `conv_err`: convergence history (`|ε²_old - ε²_new|` per iteration)
- `r_orig`, `r²`, `ρ`, `AARD`, `t`, `itercount`, `accuracy`: diagnostics
"""
Base.@kwdef @concrete struct ACEres
    X::AbstractArray
    Y::AbstractArray
    Φ_x::AbstractArray
    Θ_y::AbstractArray
    sIx::AbstractArray
    sIy::AbstractArray
    bsIx::AbstractArray
    bsIy::AbstractArray
    conv_err::AbstractArray
    plot_view_bounds::AbstractArray = [
        [(minimum(X), maximum(X)), (minimum(Y), maximum(Y))],
        [(minimum(X), maximum(X)), (minimum(Φ_x), maximum(Φ_x))],
        [(minimum(Y), maximum(Y)), (minimum(Θ_y), maximum(Θ_y))],
        [(minimum(Φ_x), maximum(Φ_x)), (minimum(Θ_y), maximum(Θ_y))],
    ]
    plot_fcs::Vector = []
    scale_factors::Vector = [zeros(2), zeros(2)]
    r_orig::AbstractArray
    r²::AbstractArray
    ρ::Float64
    AARD::Float64
    t::Float64
    itercount::Int64
    accuracy::Float64
    description::String = "ACE_simulation_result"
end

"""
    ACEsim(X, Y, smoother; errorbound, itermax_inner, itermax_outer, multiloopversion)

Description of an ACE simulation: data `X` (an `N × m` matrix) and `Y` (an
`N × 1` matrix or vector of length `N`), a smoother from
[`LocalSmoothers`](@ref) (or a vector of smoothers applied in sequence), and
convergence controls.
"""
Base.@kwdef @concrete struct ACEsim{T,S<:AbstractArray}
    X::S
    Y::S
    smoother::T
    errorbound::Float64
    itermax_inner::Int64
    itermax_outer::Int64
    multiloopversion::Symbol
end

@stable function ACEsim(X::S, Y::S, smoother::T;
                errorbound::Float64 = 1e-4,
                itermax_inner::Int64 = 50,
                itermax_outer::Int64 = 500,
                multiloopversion::Symbol = :fresh) where {T,S<:AbstractArray}
    ACEsim{T,S}(X, Y, smoother, errorbound, itermax_inner, itermax_outer, multiloopversion)
end
