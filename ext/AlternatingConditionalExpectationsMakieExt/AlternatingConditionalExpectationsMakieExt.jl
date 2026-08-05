# Makie backend for AlternatingConditionalExpectations.
#
# Loads automatically once a Makie backend (CairoMakie, GLMakie, ...) is
# imported. Defines the real methods of `plot_ace_results` and
# `benchmark_ace_plot`, which the parent module declares as stubs.

module AlternatingConditionalExpectationsMakieExt

using AlternatingConditionalExpectations
import AlternatingConditionalExpectations: ACEres, BenchmarkFunction, ε²
using Makie
using LaTeXStrings
using PrecompileTools

# Dark figure theme for the diagnostic plots.
const _ACE_THEME = Makie.Theme(
    backgroundcolor = RGBf(0.13, 0.14, 0.17),
    Axis = (
        backgroundcolor = RGBf(0.13, 0.14, 0.17),
        xgridcolor = RGBf(0.27, 0.27, 0.30),
        ygridcolor = RGBf(0.27, 0.27, 0.30),
        bottomspinecolor = :silver, leftspinecolor = :silver,
        rightspinecolor = :silver, topspinecolor = :silver,
        xtickcolor = :silver, ytickcolor = :silver,
        xlabelcolor = :white, ylabelcolor = :white, titlecolor = :white,
    ),
    Label = (color = :white,),
    Legend = (backgroundcolor = RGBf(0.18, 0.18, 0.21), framecolor = :silver,
              labelcolor = :white, titlecolor = :white),
)

# Scatter colour per diagnostic view.
const _ROW_COLORS = (:skyblue, :orange, :seagreen)

"""
    plot_ace_results(res::ACEres; savepath=nothing, figsize=(1200,800)) -> Makie.Figure

Diagnostic grid for an ACE/AVAS fit: per predictor, `Xⱼ`–`Y`, `Xⱼ`–`Φ(Xⱼ)` and
`Φ(Xⱼ)`–`Θ(Y)` scatter; plus `Y`–`Θ(Y)` and the `ε²` convergence history. The
title reports the final unexplained fraction `ε²`. Returns the `Figure`; pass
`savepath` to also write it (PNG/PDF/SVG).
"""
function AlternatingConditionalExpectations.plot_ace_results(
        res::ACEres; savepath::Union{Nothing,String}=nothing,
        figsize=(1200, 800), markersize=4)
    X, Y = res.X, vec(res.Y)
    Φx, Θy = res.Φ_x, vec(res.Θ_y)
    n = size(X, 2)
    e2 = round(abs(ε²(res.Φ_x[res.sIx], res.Θ_y[res.sIy])); digits = 4)

    fig = with_theme(_ACE_THEME) do
        f = Figure(size = figsize)
        Label(f[1, 1:2], L"ACE result:  \varepsilon^2 = %$e2";
              fontsize = 20, tellwidth = false, justification = :center)

        for j in 1:n
            xj = vec(view(X, :, j)); φj = vec(view(Φx, :, j))

            ax1 = Axis(f[2, j]; xlabel = L"X_{%$j}", ylabel = "Y")
            scatter!(ax1, xj, Y; color = _ROW_COLORS[1], markersize)

            ax2 = Axis(f[3, j]; xlabel = L"X_{%$j}", ylabel = L"\Phi(X_{%$j})")
            scatter!(ax2, xj, φj; color = _ROW_COLORS[2], markersize)

            ax3 = Axis(f[4, j]; xlabel = L"\Phi(X_{%$j})", ylabel = L"\Theta(Y)")
            scatter!(ax3, φj, Θy; color = _ROW_COLORS[3], markersize)
        end

        axY = Axis(f[5, 1]; xlabel = "Y", ylabel = L"\Theta(Y)")
        scatter!(axY, Y, Θy; color = :crimson, markersize)

        axc = Axis(f[5, 2]; xlabel = "Iterations", ylabel = L"\varepsilon")
        scatter!(axc, eachindex(res.conv_err), res.conv_err; color = :violet, markersize)

        rowgap!(f.layout, 8)
        f
    end

    isnothing(savepath) || Makie.save(savepath, fig)
    return fig
end

"""
    benchmark_ace_plot(bf::BenchmarkFunction; savepath=nothing, figsize=(900,600)) -> Makie.Figure

Scatter the benchmark samples and overlay the reference transform curves stored
in `bf.plot_fcs`.
"""
function AlternatingConditionalExpectations.benchmark_ace_plot(
        bf::BenchmarkFunction; savepath::Union{Nothing,String}=nothing,
        figsize=(900, 600))
    X = vec(bf.X); Y = vec(bf.Y)
    fig = with_theme(_ACE_THEME) do
        f = Figure(size = figsize)
        ax = Axis(f[1, 1]; xlabel = "X", ylabel = "Y", title = string(bf.name))
        scatter!(ax, X, Y; color = (:white, 0.5), markersize = 4)
        if !isempty(bf.plot_fcs)
            xs = sort(X)
            for (nm, f) in bf.plot_fcs[1]
                lines!(ax, xs, f.(xs); label = string(nm), linewidth = 2)
            end
            axislegend(ax; framecolor = :silver)
        end
        f
    end
    isnothing(savepath) || Makie.save(savepath, fig)
    return fig
end

# Precompile the figure builder so the first real plot is not paid at call time.
@compile_workload begin
    X, Y = AlternatingConditionalExpectations.generate_bivariate_data(Float64, 60, 0.1, 1.0, 42)
    res = AlternatingConditionalExpectations.ace_run(
        AlternatingConditionalExpectations.ACEsim(X, Y, AlternatingConditionalExpectations.LASb(6)))
    AlternatingConditionalExpectations.plot_ace_results(res)
end

end # module
