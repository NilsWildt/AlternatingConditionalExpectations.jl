using DrWatson
# @quickactivate "AlternatingConditionalExpectation"
include(srcdir("Kernelregression", "Kernelregression.jl"))
using .Kernelregression
include(srcdir("utils.jl"))
include(srcdir("benchmark_functions.jl"))
using AlternatingConditionalExpectation
using Statistics
using Test
# ENV["JULIA_DEBUG"] = "all"


@testset "All tests:" begin

    @testset "utils.jl" begin

# Inplace
        a = [1,5,2,8]
        b = [1,2,3,4]
        sort_two_arrays!(a, b)
        @test a == [1,2,5,8] &&  b == [1,3,2,4]

    # Not in place
        a = Array(100:-1:1)
        b = Array(1:100)
        @test sort_two_arrays(a, b)[1] == Array(1:100) &&  sort_two_arrays(a, b)[2] ==  Array(100:-1:1)


        a = Array{Float64}(100:-1:1)
        b = Array{Float64}(1:100)
        @test sort_two_arrays_native(a, b)[1] == Array(1:100) &&  sort_two_arrays_native(a, b)[2] ==  Array(100:-1:1)


        X = Array([1,2,3,4])
        @test prepend_one(X) == Array([1. 1.; 1. 2.; 1. 3.; 1. 4.])




        v = [2,3,1]
        @test rank1D(v) == [3,1,2]
        @inferred rank1D(v)     

        v = Array([[1000,10,500,15]  [1000,10,500,15]  [1000,10,500,1500]])
        @test rank(v, (1, 3)) == Float64.([[2,4,3,1] [1000,10,500,15] [2,3,1,4]])


        a = [1,5,2,8]
        asorted = [1,2,5,8]
        sind, bsind =  AlternatingConditionalExpectation.get_sort_idx_two_way(a)
        @test a[sind] == asorted
        @test asorted[bsind] == a
    end



    @testset "Kernelregression" begin
        A = ones(3, 3)
        B = .5 .* ones(3, 3)
        @test Kernelregression.pDist2(A, B) ≈ B atol = 1E-7
        a = [1,2,3,4]
        b = [1,2,3,4]
        @test Kernelregression.kernel_dot(a, b) ≈ Array([1. 2. 3. 4.; 2. 4. 6. 8.; 3. 6. 9. 12.; 4. 8. 12. 16.]) 
    end


    @testset "Module AlternatingConditionalExpectation" begin
        X, Y = AlternatingConditionalExpectation.generate_bivariate_data(200, 1, 1234) # Set seed
        @test length(X) == 200
        @test isapprox(std(X), 1.0, atol = 1e-1)

        X = rand(1000, )
        Xout = AlternatingConditionalExpectation.stoch_normalize(X)
        @test mean(X) != mean(Xout)
        @test isapprox(mean(Xout), 0, atol = 1e-12) 
        @test isapprox(var(Xout), 1)
    end


# Unittest
    @testset "benchmarks.jl" begin 
        @test size(normal_sample(15, 25)) == (15, 25)
        @test size(uniform_sample(15, 25, 0, 1)) == (15, 25)
        @inferred normal_sample(15, 25)
    end

# ################################ TESTS ###############
# begin
#     # Visual test for the LAS Smoother
#         N = 300
#         x = collect(LinRange(0, 2 * pi, N))
#         y = cos.(x) .+ 0.01 .* randn(N)
#         k = Int64(floor((N + 2) / 4))
#         # k = 10
#         @info "Gives" k
#         scatter(x, y, markersize = 1, edgecolor = :white)
#         y_LAS = LAS(x, y, k)
#         scatter!(x, y_LAS, markersize = 1, edgecolor = :white)
#         y_LAS_bound = LAS_boundary(x, y, k)
#         scatter!(x, y_LAS_bound, markersize = 1, edgecolor = :white)
#         y_LLSS = LLSS(x, y, k)
#         scatter!(x, y_LLSS, markersize = 1, edgecolor = :white)
#         Y_LLSS_bound = LLSS_boundary(x, y, k)
#         scatter!(x, Y_LLSS_bound, markersize = 1, edgecolor = :white)
#     end"


end