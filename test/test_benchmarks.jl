using ACE
using Test

@testitem "samplers" begin
    @test size(ACE.normal_sample(15, 25, 0.0, 1.0)) == (15, 25)
    @test size(ACE.uniform_sample(15, 25, 0.0, false)) == (15, 25)
    Xu = ACE.get_sample(50, 1, "uniform", 1.0, true, 7)
    @test all(0 .<= vec(Xu) .<= 1)
    Xn = ACE.get_sample(50, 1, "normal", 1.0, true, 7)
    @test all(isfinite, Xn)
end

@testitem "benchmark constructors" begin
    for F in (ACE.f_b1, ACE.f_b2, ACE.f_b3, ACE.f_b4, ACE.f_toy)
        bf = F(200, 1, "uniform", 1.0, 0.3, true, 42, false, (-5.0, 1.4))
        @test bf isa ACE.BenchmarkFunction
        @test size(bf.X) == (200, 1)
        @test size(bf.Y) == (200, 1)
        @test all(isfinite, bf.Y)
        @test !isempty(string(bf.name))
    end
end

@testitem "scaled benchmark data" begin
    bf = ACE.f_b1(200, 1, "uniform", 1.0, 0.3, true, 42, true, (-5.0, 1.4))
    @test all(0 .<= vec(bf.X) .<= 1)
    @test all(0 .<= vec(bf.Y) .<= 1)
end

@testitem "benchmark run end-to-end" begin
    bf = ACE.f_b1(300, 1, "uniform", 1.0, 0.3, true, 42, false)
    res = ACE.run(ACE.ACEsim(Matrix(bf.X), Matrix(bf.Y), ACE.LASb(20)))
    @test all(isfinite, res.Φ_x)
    @test res.ρ < 0.6
end

@testitem "all benchmarks construct (scaled) and run" begin
    for F in (ACE.f_b1, ACE.f_b2, ACE.f_b3, ACE.f_b4, ACE.f_toy)
        # scale_data=true exercises the normalization branches
        bf = F(200, 1, "uniform", 1.0, 0.3, true, 42, true, (-5.0, 1.4))
        @test all(0 .<= vec(bf.X) .<= 1)
        @test all(0 .<= vec(bf.Y) .<= 1)
        res = ACE.run(ACE.ACEsim(Matrix(bf.X), Matrix(bf.Y), ACE.LASb(15)))
        @test all(isfinite, res.Φ_x)
        @test all(isfinite, res.Θ_y)
    end
end
