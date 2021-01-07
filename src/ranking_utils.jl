"""
*Provides:* Return a vector of ranks of the given Array, along the given dimension
starting from  the smallest element.

*Use:* $(SIGNATURES)

*More specific:*
$(TYPEDSIGNATURES)
"""
function rank1D(data::AbstractArray, dim::Int64 = 1)::AbstractArray
    perm =  sortperm(data[:,dim]) 
    return perm
end

"""
*Provides:* Return a matrix, replacing the given dimensions in the orginal
array replaced by it's rank.


*Use:* $(SIGNATURES)

*More specific:*
$(TYPEDSIGNATURES)
"""
function rank(data, dimTuple = (1))
    if ! issorted(data[:,1]) 
        @info "Data wasn't sorted, will do now."
        data = hcat(sort_two_arrays(data[:,1], data[:,2]))
    end

    sortedData = copy(data)
    @info "dbg" dimTuple size(dimTuple)
    for dim in dimTuple
        perm = rank1D(sortedData[:,dim])   
        sortedData[:,dim] .= perm     
    end   
    return sortedData
end

"""
*Provides:* Return a matrix, replacing the given dimensions in the orginal
array replaced by it's rank.


*Use:* $(SIGNATURES)

*More specific:*
$(TYPEDSIGNATURES)
"""
function rank(P1::AbstractArray, P2::AbstractArray, dimTuple = (1))::AbstractArray
    if ! issorted(P1) 
        @info "Data wasn't sorted, will do now."
        P1, P2 = sort_two_arrays(P1, P2)
    end

    data = hcat(P1, P2)
    sortedData = copy(data)
    for dim in dimTuple       
        perm = rank1D(sortedData[:,dim])   
        sortedData[:,dim] .= perm     
    end   
    return sortedData
end


function calculate_empircal_2D_copula(P1::AbstractArray, P2::AbstractArray,  rankdims = (1, 2), draw = false)
    X_trans =     rank(P1, P2, rankdims)
    P1 = X_trans[:,1] ./ numSamples
    P2 = X_trans[:,2] ./ numSamples
    if draw
        X = hcat(P1, P2)
        cornerplot(X, compact = true, grid = true, markercolor = :viridis, linecolor = false)
        return P1, P2, current()
    end
    return P1, P2
end



function calculate_empircal_2D_copula(X::AbstractArray,  rankdims = (1, 2), draw = false)
    X_trans =     rank(X[:,1], X[:,2], rankdims)
    P1 = X_trans[:,1] ./ numSamples
    P2 = X_trans[:,2] ./ numSamples
    X = hcat(P1, P2)
    if draw
        cornerplot(X, compact = true, grid = true, markercolor = :viridis, linecolor = false)
        return X, current()
    end
    return X
end

