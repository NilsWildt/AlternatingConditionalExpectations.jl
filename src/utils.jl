using Documenter
using DocStringExtensions

include("error_utils.jl")
include("plot_utils.jl")
include("ranking_utils.jl")

# Simple utility functions:

add_dim(x::AbstractArray{T,N}) where {T,N} = reshape(x, Val(N + 1))

@inline function prepend_one(X::AbstractArray)
    X = cat(ones(size(X, 1)), X;dims = 2)
end


"""
*Provides:* $(FUNCTIONNAME)Return a vector of ranks of the given Array, along the given dimension
starting from  the smallest element.

*Use:* $(SIGNATURES)

*More specific:*
$(TYPEDSIGNATURES)
"""
function heaviside(x)
    return @.  0.5 * (sign(x) + 1.0)
end

## Sorting
# https://discourse.julialang.org/t/how-to-sort-two-or-more-lists-at-once/12073/13
struct CoSorterElement{T1,T2}
    x::T1
    y::T2
end
struct CoSorter{T1,T2,S <: AbstractArray{T1},C <: AbstractArray{T2}} <: AbstractVector{CoSorterElement{T1,T2}}
    sortarray::S
    coarray::C
end

Base.size(c::CoSorter) = size(c.sortarray)
Base.getindex(c::CoSorter, i...) = 
    CoSorterElement(getindex(c.sortarray, i...), getindex(c.coarray, i...))
Base.setindex!(c::CoSorter, t::CoSorterElement, i...) = 
    (setindex!(c.sortarray, t.x, i...); setindex!(c.coarray, t.y, i...); c) 
Base.isless(a::CoSorterElement, b::CoSorterElement) = isless(a.x, b.x)
Base.Sort.defalg(v::C) where {T <: Union{Number,Missing},C <: CoSorter{T}} = 
    Base.DEFAULT_UNSTABLE

function sort_two_arrays!(x::Array, y::Array)
    T = CoSorter(x, y)
    sort!(T)
    x = T.sortarray
    y = T.coarray
end

function sort_two_arrays(x::StaticArray, y::StaticArray)
    xout = zeros(Float64, size(x))
    yout = zeros(Float64, size(y))
    T = CoSorter(deepcopy(x), deepcopy(y))
    sort!(T)
    xout = T.sortarray
    yout = T.coarray
    return xout, yout
end

function sort_two_arrays_native(x::AbstractArray, y::AbstractArray)::Tuple{AbstractArray,AbstractArray}
    A = cat(x, y;dims = 2)
    B =   mapslices(sortperm, A; dims = 1)
    sortind = B[:,1]
    return x[sortind], y[sortind]
end



function lin_reg(x, y)
    β = [ ones(length(x))  x ] \ y
    β_0 = β[1]
    β = β[2:end]
    fxnew(xnew) =  β  .* xnew .+ β_0
    return fxnew
end


function logrange(start, ende, npoints)
    return  LinRange(10.0.^start, 10.0.^ende, npoints)
end