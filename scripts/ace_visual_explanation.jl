# Visual explanation of ACE as a *projection*, in 3D.
#
# ACE takes a nonlinear relationship Y = exp(sin X) and finds transforms
# Φ(X), Θ(Y) so that Θ(Y) ≈ Φ(X) — the curved relationship becomes a line.
# The construction reads off the faces of a box (axes X, Φ(X), Θ(Y)):
#   1. the raw data on the front face,
#   2. its conditional-expectation smooth E[Y|X],
#   3. the transform Φ(X) laid flat on the floor,
#   4. an additive surface raised vertically from that floor curve,
#   5. the smooth projected onto the surface at height Θ(Y),
#   6. and, read on the right face, the linearized Θ(Y)–Φ(X) line.
# This projection reading of the algorithm is due to Clara M. J. J. Roth.
#
# Run:   julia --project=<env> scripts/ace_visual_explanation.jl
# Needs: CairoMakie + MakieBake (unregistered):
#        MakieBake = {url = "https://github.com/JuliaAPlavin/MakieBake.jl"}
# Writes: assets/ace_visual_explanation/index.html (+ block PNGs, rotation.gif).

using AlternatingConditionalExpectations
using LocalSmoothers: LASb, do_smoothing
using CairoMakie
using MakieBake
using MakieBake: @o
using Random
using Statistics

const OUT = normpath(joinpath(@__DIR__, "..", "assets", "ace_visual_explanation"))
mkpath(OUT)

standardize(v) = (v .- mean(v)) ./ std(v)

# ---- data + a real ACE fit ---------------------------------------------------
rng = MersenneTwister(7)
N = 450
X = 1.7 .* randn(rng, N)
Y = exp.(sin.(X)) .+ 0.04 .* randn(rng, N)
model = ace(reshape(X, N, 1), reshape(Y, N, 1); smoother = LASb(25))

# Standardize the transforms so depth (Φ) and height (Θ) share a clean scale;
# the raw data and its smooth share the response scale of Y.
φ = standardize(vec(model.tx))          # Φ(X), used as the depth axis
θ = standardize(vec(model.ty))          # Θ(Y), used as the height axis
zY = standardize(Y)                     # raw response on the same height scale

perm = sortperm(X)
xs = X[perm]
φs = φ[perm]
θs = θ[perm]
# Conditional-expectation smooth E[Y|X]; put it on the standardized Y scale.
ys_smooth = do_smoothing(xs, Y[perm], LASb(25))
zsm = (ys_smooth .- mean(Y)) ./ std(Y)

# ---- box geometry ------------------------------------------------------------
# x = X (width), y = Φ(X) (depth), z = Y / Θ(Y) (height).
pad = 0.35
xmin, xmax = minimum(X) - pad, maximum(X) + pad
ymin, ymax = minimum(φ) - pad, maximum(φ) + pad
zmin, zmax = minimum(vcat(zY, θ, zsm)) - pad, maximum(vcat(zY, θ, zsm)) + pad
yfront = ymin                            # front face: where raw data + smooth live
zfloor = zmin                            # floor: where Φ(X) is laid flat
xright = xmax                            # right face: where Θ(Y) vs Φ(X) is read

# ---- point sets --------------------------------------------------------------
raw_pts    = Point3f.(X, yfront, zY)                       # 1: raw data, front face
smooth_pts = Point3f.(xs, yfront, zsm)                     # 2: smooth, front face
floor_pts  = Point3f.(xs, φs, zfloor)                      # 3: Φ(X) on the floor
surf_lo    = Point3f.(xs, φs, zmin)                        # 4: additive surface (band)
surf_hi    = Point3f.(xs, φs, zmax)
onsurf_pts = Point3f.(X, φ, θ)                             # 5: smooth projected up
pφ         = sortperm(φ)
wall_line  = Point3f.(xright, φ[pφ], θ[pφ])                # 6: Θ(Y)–Φ(X) on right face
wall_pts   = Point3f.(xright, φ, θ)

# A few faint connectors make the projections legible without cluttering the
# scene; keep them sparse (every ~20th point).
step = 20
seg(a, b) = reduce(vcat, [[a[i], b[i]] for i in 1:step:length(a)])
project_segs = seg(onsurf_pts, wall_pts)                  # surface → right wall

titles = (
    "1. The data:  Y = exp(sin X)",
    "2. Smooth it:  E[Y | X]",
    "3. Lay Φ(X) flat on the floor",
    "4. Raise the additive surface over Φ(X)",
    "5. Project the smooth onto the surface:  Θ(Y)",
    "6. Read Θ(Y) vs Φ(X) on the right wall",
)

# ---- build the staged, rotatable scene --------------------------------------
params = Observable((stage = 1, view = 1.30 * π))

function draw_scene!(ax, stage)
    band!(ax, surf_lo, surf_hi; color = (:dodgerblue, 0.08),
          visible = @lift($stage >= 4))
    lines!(ax, floor_pts; color = :orange, linewidth = 3,
           visible = @lift($stage >= 3))
    scatter!(ax, raw_pts; color = (:steelblue, 0.65), markersize = 6,
             visible = @lift($stage >= 1))
    lines!(ax, smooth_pts; color = :black, linewidth = 3,
           visible = @lift($stage >= 2))
    scatter!(ax, onsurf_pts; color = (:crimson, 0.8), markersize = 7,
             visible = @lift($stage >= 5))
    linesegments!(ax, project_segs; color = (:gray, 0.35), linewidth = 0.8,
                  visible = @lift($stage >= 6))
    lines!(ax, wall_line; color = :orange, linewidth = 4,
           visible = @lift($stage >= 6))
    scatter!(ax, wall_pts; color = (:crimson, 0.85), markersize = 6,
             visible = @lift($stage >= 6))
end

fig = Figure(size = (640, 480))
ax = Axis3(fig[1, 1];
    xlabel = "X", ylabel = "Φ(X)", zlabel = "Y  /  Θ(Y)",
    title = @lift(titles[$params.stage]),
    limits = (xmin, xmax, ymin, ymax, zmin, zmax),
    protrusions = (0, 0, 0, 0), elevation = 0.30, azimuth = 1.30π)
draw_scene!(ax, @lift($params.stage))
on(params) do p
    ax.azimuth[] = p.view
end

bake_html(
    params => (
        (@o _.stage) => [1, 2, 3, 4, 5, 6],
        (@o _.view)  => collect(range(1.30π; stop = 1.30π + 2π, length = 16)),
    );
    blocks = [fig],
    outdir = OUT,
)

# ---- customize the baked viewer ---------------------------------------------
# ACE title/header (not MakieBake), centered layout, and an explanation below.
write(joinpath(OUT, "layout.js"), """
const TITLE = "ACE — a visual explanation";
const HEADER = "ACE — a visual explanation";
const MAXWIDTH = "1000px";
""")

let html = read(joinpath(OUT, "index.html"), String)
    html = replace(html, r"<title>.*?</title>" => "<title>ACE — a visual explanation</title>")
    explain = """
      <section id="explain" style="max-width:760px;margin:28px auto 0;line-height:1.55;color:#222;font-family:system-ui,sans-serif;">
        <p>ACE (Breiman &amp; Friedman, 1985) estimates transformations of the
        response <em>Y</em> and the predictors <em>X</em> so that an additive
        model fits as tightly as possible:</p>
        <p style="text-align:center;font-size:1.25em;margin:1em 0;">Θ(Y) &nbsp;≈&nbsp; Σ<sub>j</sub> Φ<sub>j</sub>(X<sub>j</sub>)</p>
        <p>For the single-predictor example <em>Y</em> = exp(sin <em>X</em>), the
        slider walks through the geometry:</p>
        <ol>
          <li>the raw data on the front face;</li>
          <li>its conditional-expectation smooth E[<em>Y</em>&nbsp;|&nbsp;<em>X</em>];</li>
          <li>the predictor transform Φ(<em>X</em>) laid flat on the floor;</li>
          <li>an additive surface raised vertically over that floor curve;</li>
          <li>the smooth projected onto the surface at height Θ(<em>Y</em>);</li>
          <li>and, read on the right face, the linearized Θ(<em>Y</em>)–Φ(<em>X</em>) line.</li>
        </ol>
        <p>The projection reading of the algorithm — data onto its smooth, then
        onto the additive surface — is due to Clara M. J. J. Roth.</p>
      </section>
    """
    html = replace(html, "<footer" => explain * "  <footer")
    write(joinpath(OUT, "index.html"), html)
end

# ---- rotating GIF of the final composed scene (README embed) ----------------
let giffig = Figure(size = (640, 480))
    gifax = Axis3(giffig[1, 1];
        xlabel = "X", ylabel = "Φ(X)", zlabel = "Y  /  Θ(Y)",
        title = "ACE: from data to the additive surface  Θ(Y) ≈ Φ(X)",
        limits = (xmin, xmax, ymin, ymax, zmin, zmax),
        elevation = 0.30)
    draw_scene!(gifax, Observable(6))     # all stages composed
    record(giffig, joinpath(OUT, "rotation.gif"),
           range(1.30π; stop = 1.30π + 2π, length = 48); framerate = 12) do az
        gifax.azimuth[] = az
    end
end

println("wrote visualization to: ", OUT)
