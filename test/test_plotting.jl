using AlternatingConditionalExpectations
using Test

@testitem "plot_ace_results returns a Figure and saves a PNG" begin
    using CairoMakie
    X, Y = AlternatingConditionalExpectations.generate_bivariate_data(Float64, 150, 0.1, 1.0, 42)
    res = AlternatingConditionalExpectations.ace_run(AlternatingConditionalExpectations.ACEsim(X, Y, AlternatingConditionalExpectations.LASb(12)))
    p = plot_ace_results(res)
    @test p isa Figure
    path = tempname() * ".png"
    plot_ace_results(res; savepath = path)
    @test isfile(path)
    @test filesize(path) > 0
end

@testitem "benchmark_ace_plot returns a Figure" begin
    using CairoMakie
    bf = AlternatingConditionalExpectations.f_b1(150, 1, "uniform", 1.0, 0.3, true, 42, false)
    p = benchmark_ace_plot(bf)
    @test p isa Figure
end
