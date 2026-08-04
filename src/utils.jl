# Small utility functions (ported from the original ACE.jl).

"""
    add_dim(x::AbstractArray{T,N}) where {T,N}

Add an extra dimension to the input array.
"""
add_dim(x::AbstractArray{T,N}) where {T,N} = reshape(x, Val(N + 1))

"""
    prepend_one(X::AbstractArray)

Prepend a column of ones to the input array.
"""
@inline function prepend_one(X::AbstractArray)
    cat(ones(size(X, 1)), X; dims=2)
end

"""
    heaviside(x)

Compute the Heaviside step function (0.5 at `x == 0`).
"""
function heaviside(x)
    @. 0.5 * (sign(x) + 1.0)
end

"""
    CoSorterElement{T1,T2}

A container for paired elements to be sorted together.
"""
struct CoSorterElement{T1,T2}
    x::T1
    y::T2
end

"""
    CoSorter{T1,T2,S<:AbstractArray{T1},C<:AbstractArray{T2}}

A view that sorts two arrays simultaneously based on the first.
"""
struct CoSorter{T1,T2,S<:AbstractArray{T1},C<:AbstractArray{T2}} <: AbstractVector{CoSorterElement{T1,T2}}
    sortarray::S
    coarray::C
end

Base.size(c::CoSorter) = size(c.sortarray)
Base.getindex(c::CoSorter, i...) = CoSorterElement(getindex(c.sortarray, i...), getindex(c.coarray, i...))
Base.setindex!(c::CoSorter, t::CoSorterElement, i...) = (
    setindex!(c.sortarray, t.x, i...);
    setindex!(c.coarray, t.y, i...);
    c
)
Base.isless(a::CoSorterElement, b::CoSorterElement) = isless(a.x, b.x)
Base.Sort.defalg(v::C) where {T<:Union{Number,Missing},C<:CoSorter{T}} = Base.DEFAULT_UNSTABLE

"""
    sort_two_arrays!(x::Array, y::Array)

Sort two arrays in place based on the values in the first.
"""
function sort_two_arrays!(x::Array, y::Array)
    T = CoSorter(x, y)
    sort!(T)
    x .= T.sortarray
    y .= T.coarray
    return nothing
end

"""
    sort_two_arrays(x::StaticArray, y::StaticArray)

Sort two static arrays based on the values in the first.
"""
function sort_two_arrays(x::StaticArray, y::StaticArray)
    xm = MArray(x)  # mutable copy for in-place sorting
    ym = MArray(y)
    T = CoSorter(xm, ym)
    sort!(T)
    return SArray(xm), SArray(ym)
end

"""
    sort_two_arrays_native(x::AbstractArray, y::AbstractArray)

Sort two arrays using native Julia sorting.
"""
function sort_two_arrays_native(x::AbstractArray, y::AbstractArray)::Tuple{AbstractArray,AbstractArray}
    A = cat(x, y; dims=2)
    sortind = mapslices(sortperm, A; dims=1)[:, 1]
    return x[sortind], y[sortind]
end

"""
    lin_reg(x, y)

Fit a linear regression `y ~ x` and return a prediction function.
"""
function lin_reg(x, y)
    β = [ones(length(x)) x] \ y
    β₀, β₁ = β[1], β[2:end]
    return xnew -> β₁ .* xnew .+ β₀
end

"""
    logrange(start, ende, npoints)

Create a logarithmically spaced range (base 10).
"""
function logrange(start, ende, npoints)
    LinRange(10.0^start, 10.0^ende, npoints)
end
