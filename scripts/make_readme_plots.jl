# Regenerate the images embedded in the README. Writes PNGs to ../assets/.
# Run headless: julia --project=. scripts/make_readme_plots.jl

ENV["GKSwstype"] = "100"   # headless GR backend (no display)

using AlternatingConditionalExpectations
using Plots
using Random
gr()

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
p1 = scatter(Xv, φ; title = "Φ(X): recovered predictor transform",
             xlabel = "X", ylabel = "Φ(X)", mc = :dodgerblue)
p2 = scatter(Yv, θ; title = "Θ(Y): recovered response transform",
             xlabel = "Y", ylabel = "Θ(Y)", mc = :crimson)
p3 = scatter(φ, θ; title = "Linearized:  Θ(Y) vs Φ(X)   (r² = $(round(model.rsq, digits = 3)))",
             xlabel = "Φ(X)", ylabel = "Θ(Y)", mc = :seagreen)
hero = plot(p1, p2, p3; layout = (1, 3), size = (1500, 430), ms = 3, msw = 0,
            legend = false, framestyle = :box)
savefig(hero, joinpath(ASSETS, "ace_transforms.png"))

# 2) LocalSmoothers on a noisy signal, including the super smoother.
rng = MersenneTwister(1)
x = sort(rand(rng, 400))
truth = sin.(2π .* x)
y = truth .+ 0.25 .* randn(rng, 400)
plt = scatter(x, y; label = "data", ms = 2, mc = :gray70, msw = 0,
              legend = :topright, size = (1000, 520), framestyle = :box,
              title = "LocalSmoothers on a noisy signal")
plot!(plt, x, truth; label = "truth", lw = 2, lc = :black, ls = :dash)
for (sm, lab, col) in ((LASb(40), "LASb(40)", :dodgerblue),
                       (LLSSb(40), "LLSSb(40)", :orange),
                       (Supsmu(), "Supsmu (CV span)", :crimson))
    plot!(plt, x, do_smoothing(x, y, sm); label = lab, lw = 2.5, lc = col)
end
savefig(plt, joinpath(ASSETS, "smoothers.png"))

println("wrote: ", join(readdir(ASSETS), ", "))
