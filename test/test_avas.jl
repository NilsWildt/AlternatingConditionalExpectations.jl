using ACE
using Test

@testitem "ctsub trapezoidal integral" begin
    # Constant weight → linear cumulative integral.
    u = [0.0, 1.0, 2.0]
    @test ctsub(u, [1.0, 1.0, 1.0], [0.0, 1.0, 2.0]) ≈ [0.0, 1.0, 2.0]
    @test ctsub(u, [2.0, 2.0, 2.0], [0.0, 1.0, 2.0]) ≈ [0.0, 2.0, 4.0]

    # Linear weight v(s) = 2s on [0,1] → ∫₀¹ 2s ds = 1.
    @test ctsub([0.0, 1.0], [0.0, 2.0], [1.0])[1] ≈ 1.0

    # Outside the grid: extend using the end slope.
    @test ctsub([0.0, 1.0], [1.0, 1.0], [-1.0])[1] ≈ -1.0   # below u[1]
    @test ctsub([0.0, 1.0], [1.0, 1.0], [2.0])[1] ≈ 2.0     # above u[n]

    # A strictly positive weight integrates to a monotone-increasing transform.
    y = collect(-0.5:0.25:1.5)
    @test issorted(ctsub([0.0, 0.5, 1.0], [1.0, 2.0, 3.0], y))

    @test_throws DimensionMismatch ctsub([0.0, 1.0], [1.0], [0.5])
end

@testitem "avas_run converges on bivariate data" begin
    X, Y = generate_bivariate_data(Float64, 300, 0.1, 1.0, 42)
    res = avas_run(ACEsim(X, Y, LASb(20)))
    @test res isa ACEres
    @test isfinite(res.ρ)
    @test res.ρ < 0.1
    @test res.r²[1] > 0.9
    @test res.itercount > 0
    @test all(isfinite, res.Φ_x) && all(isfinite, res.Θ_y)
end

@testitem "avas_run works across smoother types" begin
    X, Y = generate_bivariate_data(Float64, 200, 0.1, 1.0, 42)
    for sm in (LAS(15), LASb(15), LLSS(20), LLSSb(20), NWKernelsmooth(Gaussian(0.5)))
        res = avas_run(ACEsim(X, Y, sm))
        @test all(isfinite, res.Φ_x)
        @test all(isfinite, res.Θ_y)
        @test isfinite(res.ρ)
    end
end

@testitem "avas_run on multivariate data" begin
    using Random
    rng = MersenneTwister(42)
    n = 300
    X = rand(rng, n, 2)
    Y = reshape(X[:, 1] .^ 2 .+ sin.(X[:, 2]) .+ 0.1 .* randn(rng, n), n, 1)
    res = avas_run(ACEsim(X, Y, LLSSb(25)))
    @test res isa ACEres
    @test length(res.r²) == 2
    @test all(isfinite, res.r²)
    @test isfinite(res.ρ)
end

@testitem "AARD guards zero-valued truth" begin
    y = [0.0, 1.0, 2.0, 4.0]
    yhat = [0.1, 1.1, 1.8, 4.2]
    a = ACE.AARD(y, yhat)
    @test isfinite(a)                                   # no Inf from the zero entry
    @test a ≈ ACE.AARD([1.0, 2.0, 4.0], [1.1, 1.8, 4.2])  # equals AARD over nonzero terms
    @test isnan(ACE.AARD([0.0, 0.0], [1.0, 2.0]))       # all-zero truth → NaN
end
