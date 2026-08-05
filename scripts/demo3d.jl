# Multivariate ACE demo: Y = X₁² + sin(X₂) + ε
#
# Run with:  julia --project=. scripts/demo3d.jl
# Saves:     output/demo3d_ace.png
using AlternatingConditionalExpectations
using LocalSmoothers
using Plots
using Random
ENV["GKSwstype"] = "100"  # headless

outdir = joinpath(@__DIR__, "..", "output")
mkpath(outdir)

rng = MersenneTwister(42)
n = 400
X = rand(rng, n, 2)
Y = reshape(X[:, 1] .^ 2 .+ sin.(X[:, 2]) .+ 0.1 .* randn(rng, n), n, 1)

sim = ACEsim(X, Y, LLSSb(25))
res = AlternatingConditionalExpectations.ace_run(sim)

@info "Multivariate ACE result" ρ = res.ρ r² = res.r² iters = res.itercount t = res.t

path = joinpath(outdir, "demo3d_ace.png")
plot_ace_results(res; savepath = path)
println("saved -> ", path)
