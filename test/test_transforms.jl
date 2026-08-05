using ACE
using Test

@testitem "isotonic regression (PAVA)" begin
    @test ACE.isotonic([1.0, 3.0, 2.0, 4.0]) ≈ [1.0, 2.5, 2.5, 4.0]
    @test ACE.isotonic([4.0, 2.0, 3.0, 1.0]) ≈ [4.0, 2.5, 2.5, 1.0]   # decreasing fit chosen
    @test ACE.isotonic([1.0, 2.0, 3.0]) ≈ [1.0, 2.0, 3.0]             # already monotone
    p = ACE.isotonic([3.0, 1.0, 2.0, 5.0, 4.0])
    @test issorted(p) || issorted(p; rev=true)
    @test ACE.isotonic([2.0]) == [2.0]
end

@testitem "transform_fit variants" begin
    using Random
    rng = MersenneTwister(3)
    x = collect(range(0, 1, length=50))
    sidx = sortperm(x)
    bidx = invperm(sidx)

    # LinearFit recovers a line
    lf = ACE.transform_fit(LinearFit(), x, 2 .* x .+ 1 .+ 0.001 .* randn(rng, 50), sidx, bidx)
    @test lf ≈ (2 .* x .+ 1) atol = 0.05

    # Categorical maps each level to its group mean
    xc = repeat([1.0, 2.0, 3.0]; inner=10)
    tc = repeat([10.0, 20.0, 30.0]; inner=10)
    sc = sortperm(xc)
    bc = invperm(sc)
    @test ACE.transform_fit(Categorical(), xc, tc, sc, bc) ≈ tc

    # Monotone output is monotone in ascending-x order
    mono = ACE.transform_fit(Monotone(LASb(8)), x, sin.(2π .* x), sidx, bidx)
    ms = mono[sidx]
    @test issorted(ms) || issorted(ms; rev=true)

    # Periodic smoothing stays finite over a full period
    xp = collect(range(0, 2π, length=60))
    sp = sortperm(xp)
    bp = invperm(sp)
    per = ACE.transform_fit(Periodic(LASb(6), 2π), xp, sin.(xp), sp, bp)
    @test all(isfinite, per) && length(per) == 60
end
