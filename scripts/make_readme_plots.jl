# Regenerate the images embedded in the README. Writes PNGs to ../assets/.
# Run:  julia --project=. scripts/make_readme_plots.jl
# Requires CairoMakie + Makie in the active environment:
#   julia --project=. -e 'import Pkg; Pkg.add(["CairoMakie","Makie"])'

using AlternatingConditionalExpectations
using CairoMakie
using Random

const ASSETS = normpath(joinpath(@__DIR__, "..", "assets"))
mkpath(ASSETS)

# 1) What ACE does: given Y = exp(sin X) it recovers the nonlinear predictor
# transform Φ(X) ≈ sin(X) and response transform Θ(Y) ≈ log(Y), which together
# linearize the relationship (Θ(Y) ≈ Φ(X)).
rng0 = MersenneTwister(7)
n0 = 300
Xv = 1.7 .* randn(rng0, n0)
Yv = exp.(sin.(Xv)) .+ 0.03 .* randn(rng0, n0)
model = ace(reshape(Xv, n0, 1), reshape(Yv, n0, 1); smoother = LASb(20))
φ = vec(model.tx)
θ = vec(model.ty)

hero = Figure(size = (1500, 430))
ax1 = Axis(hero[1, 1]; title = "Φ(X): recovered predictor transform",
           xlabel = "X", ylabel = "Φ(X)")
scatter!(ax1, Xv, φ; color = :dodgerblue, markersize = 7)
ax2 = Axis(hero[1, 2]; title = "Θ(Y): recovered response transform",
           xlabel = "Y", ylabel = "Θ(Y)")
scatter!(ax2, Yv, θ; color = :crimson, markersize = 7)
ax3 = Axis(hero[1, 3];
           title = "Linearized:  Θ(Y) vs Φ(X)   (r² = $(round(model.rsq, digits = 3)))",
           xlabel = "Φ(X)", ylabel = "Θ(Y)")
scatter!(ax3, φ, θ; color = :seagreen, markersize = 7)
save(joinpath(ASSETS, "ace_transforms.png"), hero)

# 2) LocalSmoothers on a noisy signal, including the super smoother.
rng = MersenneTwister(1)
x = sort(rand(rng, 400))
truth = sin.(2π .* x)
y = truth .+ 0.25 .* randn(rng, 400)

smoothers = Figure(size = (1000, 520))
ax = Axis(smoothers[1, 1]; title = "LocalSmoothers on a noisy signal",
          xlabel = "x", ylabel = "y")
scatter!(ax, x, y; color = :gray70, markersize = 5, label = "data")
lines!(ax, x, truth; color = :black, linestyle = :dash, linewidth = 2, label = "truth")
for (sm, lab, col) in ((LASb(40), "LASb(40)", :dodgerblue),
                       (LLSSb(40), "LLSSb(40)", :orange),
                       (Supsmu(), "Supsmu (CV span)", :crimson))
    lines!(ax, x, do_smoothing(x, y, sm); color = col, linewidth = 2.5, label = lab)
end
axislegend(ax; position = :rt)
save(joinpath(ASSETS, "smoothers.png"), smoothers)

println("wrote: ", join(readdir(ASSETS), ", "))
