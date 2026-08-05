# Bivariate ACE demo.
#
# Run with:  julia --project=. scripts/demo2d.jl
# Saves:     output/demo2d_ace.png
using AlternatingConditionalExpectations
using LocalSmoothers
using CairoMakie

outdir = joinpath(@__DIR__, "..", "output")
mkpath(outdir)

# Data: Y = exp(X³ + ε)
X, Y = AlternatingConditionalExpectations.generate_bivariate_data(Float64, 500, 0.1, 1.0, 42)
sim = ACEsim(X, Y, LASb(20))
res = AlternatingConditionalExpectations.ace_run(sim)

@info "ACE result" ρ = res.ρ r² = res.r²[1] iters = res.itercount t = res.t

path = joinpath(outdir, "demo2d_ace.png")
plot_ace_results(res; savepath = path)
println("saved -> ", path)
