# Smoother comparison demo.
#
# Run with:  julia --project=. scripts/demoSmooth2d.jl
# Saves:     output/demosmooth2d_comparison.png
using AlternatingConditionalExpectations
using LocalSmoothers
using CairoMakie

outdir = joinpath(@__DIR__, "..", "output")
mkpath(outdir)

n = 400
x = collect(range(0.0, 4.0; length = n))
y = sin.(x) .+ 0.2 .* randn(n)

fig = Figure(size = (900, 500))
ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", title = "Local smoothers compared")
scatter!(ax, x, y; color = (:black, 0.3), markersize = 4, label = "data")

smoothers = [
    "LAS"   => LAS(41),
    "LASb"  => LASb(41),
    "LLSS"  => LLSS(41),
    "LLSSb" => LLSSb(41),
    "FRSS"  => FRSS([0.05, 0.1, 0.5], 0.2, 0.2),
    "NW"    => NWKernelsmooth(Gaussian(0.3)),
]
for (label, sm) in smoothers
    ys = do_smoothing(x, y, sm)
    lines!(ax, x, ys; label = label, linewidth = 2)
end
axislegend(ax; position = :rt)

path = joinpath(outdir, "demosmooth2d_comparison.png")
save(path, fig)
println("saved -> ", path)
