using ACE
using Test

@testitem "ace() matches the legacy ace_run" begin
    X, Y = generate_bivariate_data(Float64, 300, 0.1, 1.0, 42)
    m = ace(X, Y; smoother=LASb(15))
    r = ace_run(ACEsim(X, Y, LASb(15)))
    @test m.ρ ≈ r.ρ
    @test m.rsq ≈ 1 - m.ρ
    @test m.tx === m.Φ_x && m.ty === m.Θ_y
    @test m.iters == m.itercount
    @test m.converged isa Bool
    @test :rsq in propertynames(m) && :tx in propertynames(m)
end

@testitem "ace() input forms, per-variable transforms, and errors" begin
    X, Y = generate_bivariate_data(Float64, 200, 0.1, 1.0, 7)
    @test ace(vec(X), vec(Y); smoother=LASb(12)).rsq > 0.9   # vector inputs
    @test ace(X, Y; smoother=LASb(12), xtransforms=Monotone(LASb(12))) isa ACEres
    # one transform per predictor is required
    @test_throws ArgumentError ace(X, Y; smoother=LASb(12),
                                   xtransforms=[Smooth(LASb(12)), Smooth(LASb(12))])
    @test_throws ArgumentError ace(X, Y)   # neither smoother nor transforms
end

@testitem "avas() runs and converges" begin
    X, Y = generate_bivariate_data(Float64, 300, 0.1, 1.0, 42)
    m = avas(X, Y; smoother=LASb(20))
    @test m.rsq > 0.9
    @test m.converged
end

@testitem "predict recovers the fit at training points" begin
    using Statistics
    X, Y = generate_bivariate_data(Float64, 200, 0.1, 1.0, 42)
    m = ace(X, Y; smoother=LASb(15))
    p = predict(m, X)
    fitsum = vec(sum(m.Φ_x; dims=2))
    @test length(p) == size(X, 1)
    @test cor(p, fitsum) > 0.999
    @test predict(m, vec(X)) ≈ p
    @test_throws DimensionMismatch predict(m, rand(5, 3))
end

@testitem "mixed categorical + smooth predictors" begin
    using Random
    rng = MersenneTwister(1)
    n = 200
    Xm = hcat(rand(rng, n), Float64.(rand(rng, 1:3, n)))
    Ym = reshape(Xm[:, 1] .^ 2 .+ 0.5 .* Xm[:, 2] .+ 0.05 .* randn(rng, n), n, 1)
    m = ace(Xm, Ym; smoother=LASb(15), xtransforms=[Smooth(LASb(15)), Categorical()])
    @test length(m.r²) == 2
    @test m.rsq > 0.8
end

@testitem "FitControls caps iterations" begin
    X, Y = generate_bivariate_data(Float64, 200, 0.1, 1.0, 42)
    m = ace(X, Y; smoother=LASb(15), itermax_outer=1, itermax_inner=1)
    @test m.iters ≤ 1
    @test !m.converged
end
