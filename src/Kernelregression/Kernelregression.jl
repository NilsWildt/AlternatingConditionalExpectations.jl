module Kernelregression
# Defines different kernels that can be used for regularized kernel interpolation (kernel regression)
# Using: Approximation with Kernel Methods WS 17/18 by Dr. Gabriele Santin & Prof. Haasdonk, Univ. of Stuttgart
using LinearAlgebra
using Revise
using StaticArrays
abstract type Kernel end
export get_kernel_interpolant, Kernel


include("k_gaussian.jl")
include("k_polynomial.jl")
include("k_linear.jl")
include("k_imq.jl")
include("k_mq.jl")
include("k_epanechnikov.jl")


function prepend_one(X::AbstractArray)
    X = cat(ones(size(X, 1)), X;dims = 2)
end

""" 
pDist2 
pairwise euclidean distance
```math
(x - y) = sqrt(x ^ 2 + y ^ 2 - 2 * x * y)
```
""" 
function pDist2(X::AbstractArray, Y::AbstractArray)
    Ly = size(Y, 1)
    Lx = size(X, 1)
    D = sqrt(pDist2Squared(X, Y))
    return D
end

 
function pDist2Squared(X::AbstractArray, Y::AbstractArray)
    # This implementation uses
    # (x - y) ^ 2 = x ^ 2 + y ^ 2 - 2 * x * y
    Ly = size(Y, 1)
    Lx = size(X, 1)
    T1 = sum(X.^2;dims = 2) * ones(1, Ly)
    T2 = ones(Lx, 1) * sum(Y.^2;dims = 2)' 
    T3 = - 2 .* X * Y'
    D = T1 + T2 + T3
    return D
end


function kernel_dot(X::AbstractArray, Y::AbstractArray)
    dimY = size(Y)
    dimX = size(X)
    Lx = Int64(dimX[1])
    Ly = Int64(dimY[1])
    A = zeros(Float64, (Lx, Ly))
    @inbounds @simd for i in 1:Ly 
        for j in 1:Lx
            A[i,j] = @views X[i] * Y[j]
        end
    end
    display(A)
    return A
end


function regularize!(K::AbstractMatrix, reg::Float64 = 1E-6)
    K .+= reg .* 1.0I(size(K)[1])    
end

function get_kernel_interpolant(X::AbstractArray, Y::AbstractArray, mykernel::Kernel, reg::Float64 = 1E-6)
    K = evalKmatrix(mykernel, X, X) # Both arrays (N,1)
    regularize!(K, reg)
    facts = (K \ Y) 
    fxnew(xnew) = evalKmatrix(mykernel, xnew, X) * facts
    return fxnew    
end

end # End module
