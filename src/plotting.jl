# Plotting: ACE result recipe + convenience helpers (Plots-based).

"""
    plot(res::ACEres; ...)

Plots recipe for an [`ACEres`](@ref) object: a dark-themed figure with the
data, the fitted transformations and the convergence history.
"""
@recipe function f(bf::ACEres; transform=false, full=true, dpi=500, plotsize=2.0 .* (1200, 800))
    markershape --> :circle
    markersize --> 2
    link --> :none
    size --> plotsize
    margin --> 20Plots.px

    if full
        X = bf.X
        Y = bf.Y
        Φ_x = bf.Φ_x
        Θ_y = bf.Θ_y

        plot_view_bounds = bf.plot_view_bounds
        plot_fcs = bf.plot_fcs
        n = size(X, 2)

        grid := false
        layout := @layout [a{0.05h}; grid(3, n) b{0.5w}; c{0.1h}]
        seriestype := :scatter
        background_color := RGB(0.2, 0.2, 0.2)
        dpi := dpi
        colorbar := false
        legend := false
        markerstrokewidth := 0

        e2 = string(round(abs((ε²(Φ_x[bf.sIx], Θ_y[bf.sIy]))); digits=4))
        mytit = L"ACE result: \varepsilon^2 = " * e2
        title := mytit

        # Header (subplot 1)
        @series begin
            seriestype := :scatter
            framestyle := :none
            subplot := 1
        end

        # First row (X vs Y)
        for i in 1:Int(n)
            @series begin
                seriestype := :scatter
                title := ""
                xguide := L"X_{%$i}"
                yguide := L"Y"
                subplot := i + 1
                X[:, i], Y
            end
        end

        # Second row (X vs Φ(X))
        for i in 1:Int(n)
            @series begin
                seriestype := :scatter
                title := ""
                xlabel --> L"X_{%$i}"
                ylabel --> L"\Phi(X_{%$i})"
                subplot := n + 1 + i
                X[:, i], Φ_x[:, i]
            end
        end

        # Third row (Φ(X) vs Θ(Y))
        for i in 1:Int(n)
            @series begin
                title := ""
                xlabel --> L"\Phi(X_{%$i})"
                ylabel --> L"\Theta(Y)"
                subplot := 2 * n + 1 + i
                Φ_x[:, i], Θ_y
            end
        end

        # Y vs Θ(Y) plot
        @series begin
            title := ""
            xlabel --> L"Y"
            ylabel --> L"\Theta(Y)"
            subplot := 3 * n + 1 + 1
            Y, Θ_y
        end

        # Convergence plot
        @series begin
            seriestype := :scatter
            markersize := 3
            title := ""
            xlabel --> L"Iterations"
            ylabel --> L"\varepsilon"
            subplot := 3 * n + 1 + 1 + 1
            collect(1:length(bf.conv_err)), collect(bf.conv_err)
        end

    else
        if transform && length(bf.Φ_x) != 0
            x := bf.Φ_x
            y := bf.Θ_y
        else
            x := bf.X
            y := bf.Y
        end
    end
end

"""
    plot_ace_results(res::ACEres; savepath=nothing, kwargs...)

Plot an [`ACEres`](@ref) result and optionally save it to `savepath` (PNG/PDF/SVG
via Plots). Returns the `Plots.Plot` object.
"""
function plot_ace_results(res::ACEres; savepath=nothing, kwargs...)
    p = plot(res; kwargs...)
    isnothing(savepath) || Plots.savefig(p, savepath)
    return p
end

"""
    benchmark_ace_plot(bf::BenchmarkFunction; savepath=nothing, figsize=(900, 600))

Scatter plot of the benchmark data, overlaid with the reference transform
curves stored in `bf.plot_fcs`.
"""
function benchmark_ace_plot(bf::BenchmarkFunction; savepath=nothing, figsize=(900, 600))
    X = vec(bf.X)
    Y = vec(bf.Y)
    p = scatter(X, Y; xlabel="X", ylabel="Y", label=string(bf.name),
                markersize=2, alpha=0.5, size=figsize)
    if !isempty(bf.plot_fcs)
        xs = sort(X)
        for (n, f) in bf.plot_fcs[1]
            plot!(p, xs, f.(xs); label=string(n), linewidth=2)
        end
    end
    isnothing(savepath) || Plots.savefig(p, savepath)
    return p
end
