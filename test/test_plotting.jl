using ACE
using Test

@testitem "plot_ace_results returns a plot and saves a PNG" begin
    using Plots
    ENV["GKSwstype"] = "100"  # headless GR backend
    X, Y = ACE.generate_bivariate_data(Float64, 150, 0.1, 1.0, 42)
    res = ACE.ace_run(ACE.ACEsim(X, Y, ACE.LASb(12)))
    p = plot_ace_results(res)
    @test p isa Plots.Plot
    path = tempname() * ".png"
    plot_ace_results(res; savepath = path)
    @test isfile(path)
    @test filesize(path) > 0
end

@testitem "benchmark_ace_plot returns a plot" begin
    using Plots
    ENV["GKSwstype"] = "100"
    bf = ACE.f_b1(150, 1, "uniform", 1.0, 0.3, true, 42, false)
    p = benchmark_ace_plot(bf)
    @test p isa Plots.Plot
end
