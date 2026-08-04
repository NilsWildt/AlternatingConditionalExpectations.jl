using ACE
using Test

@testitem "sort_two_arrays! (in place)" begin
    a = [1, 5, 2, 8]
    b = [1, 2, 3, 4]
    ACE.sort_two_arrays!(a, b)
    @test a == [1, 2, 5, 8]
    @test b == [1, 3, 2, 4]
end

@testitem "sort_two_arrays (static)" begin
    using StaticArrays
    a = SVector{100}(100:-1:1)
    b = SVector{100}(1:100)
    sa, sb = ACE.sort_two_arrays(a, b)
    @test sa == SVector{100}(1:100)
    @test sb == SVector{100}(100:-1:1)
end

@testitem "sort_two_arrays_native" begin
    a = collect(100.0:-1:1.0)
    b = collect(1.0:100.0)
    sa, sb = ACE.sort_two_arrays_native(a, b)
    @test sa == collect(1.0:100.0)
    @test sb == collect(100.0:-1:1.0)
end

@testitem "prepend_one" begin
    X = [1.0 2.0; 3.0 4.0]
    Xp = ACE.prepend_one(X)
    @test size(Xp) == (2, 3)
    @test all(Xp[:, 1] .== 1.0)
    @test Xp[:, 2:end] == X
end

@testitem "heaviside" begin
    @test ACE.heaviside(-2.0) == 0.0
    @test ACE.heaviside(0.0) == 0.5
    @test ACE.heaviside(3.0) == 1.0
end

@testitem "lin_reg" begin
    x = collect(range(0.0, 1.0, length=20))
    y = 3.0 .* x .+ 2.0
    f = ACE.lin_reg(x, y)
    @test only(f(0.5)) ≈ 3.5 atol = 1e-8
end

@testitem "logrange" begin
    r = ACE.logrange(0, 2, 5)
    @test length(r) == 5
    @test r[1] ≈ 1.0
    @test r[end] ≈ 100.0
end

@testitem "add_dim" begin
    x = [1.0, 2.0]
    @test size(ACE.add_dim(x)) == (2, 1)
end
