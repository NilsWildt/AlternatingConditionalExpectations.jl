using ACE
using Test

@testitem "generate_bivariate_data" begin
    using Statistics
    X, Y = generate_bivariate_data(Float64, 200, 0.1, 1.0, 42)
    @test size(X) == (200, 1)
    @test size(Y) == (200, 1)
    @test all(isfinite, X) && all(isfinite, Y)
    @test std(vec(X)) ≈ 1.0 atol = 1e-1
    X2, Y2 = generate_bivariate_data(Float64, 200, 0.1, 1.0, 42)
    @test X == X2 && Y == Y2
    X3, Y3 = generate_bivariate_data(100, 0.1, 1.0, 7)
    @test size(X3) == (100, 1)
end

@testitem "stoch_normalize" begin
    using Statistics
    v = randn(500) .* 2 .+ 5
    out = stoch_normalize(v)
    @test mean(out) ≈ 0 atol = 1e-10
    @test std(out) ≈ 1 atol = 1e-10
    c = fill(3.0, 50)
    @test all(stoch_normalize(c) .== 0.0)
end

@testitem "ε²" begin
    Φ = randn(100, 1)
    Θ = vec(sum(Φ; dims=2))
    @test ε²(Φ, Θ) ≈ 0 atol = 1e-10
    Φ2 = randn(100, 1)
    Θ2 = randn(100)
    @test ε²(Φ2, Θ2) < 5
    @test ε²(Φ2, Θ2) > 1
end

@testitem "run converges with LASb" begin
    X, Y = generate_bivariate_data(Float64, 300, 0.1, 1.0, 42)
    res = ACE.ace_run(ACEsim(X, Y, LASb(15)))
    @test res isa ACEres
    @test res.ρ < 0.05
    @test res.r²[1] > 0.9
    @test res.itercount > 0
    @test all(isfinite, res.conv_err)
    @test res.conv_err[1] > res.conv_err[end]
    @test all(isfinite, res.Φ_x) && all(isfinite, res.Θ_y)
end

@testitem "run works with all smoother types" begin
    X, Y = generate_bivariate_data(Float64, 200, 0.1, 1.0, 42)
    smoothers = [
        LAS(10), LASb(10), LLSS(10), LLSSb(10),
        FRSS([0.1, 0.3], 0.2, 0.2),
        NWKernelsmooth(Gaussian(0.5)),
        Kernelsmooth(Gaussian(0.5)),
    ]
    for sm in smoothers
        res = ACE.ace_run(ACEsim(X, Y, sm))
        @test all(isfinite, res.Φ_x)
        @test all(isfinite, res.Θ_y)
        @test isfinite(res.ρ)
    end
end

@testitem "run works with a vector of smoothers" begin
    X, Y = generate_bivariate_data(Float64, 200, 0.1, 1.0, 42)
    res = ACE.ace_run(ACEsim(X, Y, [LASb(10), LASb(12)]))
    @test all(isfinite, res.Φ_x)
    @test res.r²[1] > 0.8
end

@testitem "run on multivariate data" begin
    using Random
    rng = MersenneTwister(42)
    n = 300
    X = rand(rng, n, 2)
    Y = reshape(X[:, 1] .^ 2 .+ sin.(X[:, 2]) .+ 0.1 .* randn(rng, n), n, 1)
    res = ACE.ace_run(ACEsim(X, Y, LLSSb(25)))
    @test res isa ACEres
    @test res.ρ < 0.1
    @test length(res.r²) == 2
    @test all(isfinite, res.r²)
end

@testitem "error metrics" begin
    y = collect(range(0.1, 1.0, length=50))
    yhat = y .+ 0.01
    @test ACE.RMSE(y, yhat) ≈ 0.01 atol = 1e-12
    @test ACE.MAE(y, yhat) ≈ 0.01 atol = 1e-12
    @test ACE.pErr(y, yhat) ≈ 0.0001 atol = 1e-12
    @test ACE.myErr(y, yhat) ≈ 0.01 atol = 1e-12
    @test isfinite(ACE.AARD(y, yhat))
    @test isfinite(ACE.UFV(y, yhat))
    @test isfinite(ACE.nMAE(y, yhat))
end
