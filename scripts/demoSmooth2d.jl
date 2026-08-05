# Smoother comparison demo.
#
# Run with:  julia --project=. scripts/demoSmooth2d.jl
# Saves:     output/demosmooth2d_comparison.png
using AlternatingConditionalExpectations
using LocalSmoothers
using Plots
ENV["GKSwstype"] = "100"  # headless

outdir = joinpath(@__DIR__, "..", "output")
mkpath(outdir)

n = 400
x = collect(range(0.0, 4.0, length=n))
y = sin.(x) .+ 0.2 .* randn(n)

p = scatter(x, y; label = "data", markersize = 1.5, alpha = 0.4, size = (900, 500))
smoothers = [
    "LAS"  => LAS(41),
    "LASb" => LASb(41),
    "LLSS" => LLSS(41),
    "LLSSb" => LLSSb(41),
    "FRSS" => FRSS([0.05, 0.1, 0.5], 0.2, 0.2),
    "NW"   => NWKernelsmooth(Gaussian(0.3)),
]
for (label, sm) in smoothers
    ys = do_smoothing(x, y, sm)
    plot!(p, x, ys; label = label, linewidth = 2)
end

path = joinpath(outdir, "demosmooth2d_comparison.png")
savefig(p, path)
println("saved -> ", path)
