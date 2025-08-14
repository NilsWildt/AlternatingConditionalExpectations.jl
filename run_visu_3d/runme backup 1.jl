### A Pluto.jl notebook ###
# v0.16.1

import Pkg
Pkg.activate("/home/nilswildt/Syncthing/Documents/Arbeit/PhD/Operativ/Projekte/ACE/Presentations/2021_10_26/run_visu_3d")
# Pkg.instantiate()
using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 5567d952-1ac3-11ec-2fac-d526f986747c
begin
	using Distributions
	using Random
	using Statistics
	using Plots
	using StatsPlots
	using StatsBase
	using PrettyTables
	using PlutoUI
	using Latexify
	using Query
	using DataFrames
	using Plots.PlotMeasures 
	using GLMakie
	using Makie
	
end

begin
# ╔═╡ 5569933f-ce01-46ba-b699-2d0897c6cc88
# html"""<style>
# main {
#     max-width: 900px;
# }
# """

# ╔═╡ 76b41d24-016b-452b-9ff4-89739f93d7ff
TupList2Matrix(L) = reduce(hcat,  map(x -> getindex.(L, x), 1:2));

# ╔═╡ eb42c2d5-1768-4725-bb68-3131f8f9c4f1
RMSE(x,y) =  Statistics.mean(sqrt.((x.-y).^2));

# ╔═╡ a28683aa-e71b-4739-86e3-0edfc5e35b11
# md"""
# 	Select X sampling: normal or uniform:
	
# 	$(@bind smethod Select(["normal","uniform"]))
	
# 	"""

smethod="uniform"

# ╔═╡ 2a1f2a4f-6edd-4e45-ba80-320b77d90f80
# md"""
# 	Tradeoff between transformations:
	
# 	$(@bind α PlutoUI.Slider(0:0.05:1,show_value=true))
	
# 	"""
α = 1.0
# ╔═╡ 4a270acd-82cf-4cab-bb18-b08dc5d08118
@. θ(y) = α*(log(y))+(1-α)*asin(log(y));

# ╔═╡ 7cbfe2c8-f308-4164-8872-1e653ade631b
β=1-α

# ╔═╡ 8d6fff3f-f4cf-4e7b-8c51-3db9ca64ff71
@. Φ(x) = (1-β)*sin(x)+β*x;

# ╔═╡ c21e33af-0199-4564-9926-ce6398c53564
	# md"""
	# multiplicative error δ:
	
	# $(@bind δ PlutoUI.Slider(0:0.05:2,show_value=true))
	
	# """
δ  = 0.3	

# ╔═╡ b927d87d-a042-4ff0-a420-e1de97afd2f5
	# md"""
	# additive error γ:
	
	# $(@bind γ PlutoUI.Slider(0:0.05:1,show_value=true))
	
	# """
γ = 0.0

# ╔═╡ 5f53d05b-60c9-4d61-8814-4bc80528722f
	function sample_process()
    seed = 42
    N = 500
    rng = Random.MersenneTwister(seed)
	x = (rand(rng,N).*10).-5.0
    if smethod=="normal"
		x = 2. .*randn(rng,N)
	end
    ε = randn(rng,N)
	# ra = randn(rng,N)
    y = @. exp(sin(x+δ*ε))#+γ* ra # = exp(x^2)*exp(eps)
	z = LinRange(-5,5,N)
	s = @. exp(sin(z))
    return x,y,ε,z,s
	end
	function normalize(x)
		return normalize_mean(x)/std(x)
	end
	
	function normalize_mean(x)
		return x.-mean(x)
	end
	
	x,y,ε,z,s = sample_process()
	d = [(x[i],y[i]) for i in 1:length(x)];
	dnew = filter(x->x[2]<30,d)
	
	A = TupList2Matrix(dnew)
	x = A[:,1]
	y = A[:,2];
	# gr()
	# p1 = Plots.scatter(x,y,xlabel="x",ylabel="y",leg=false)
	# Plots.plot!(p1,z,s,lw=3)
	# p2 = marginalhist(x,y,leg=false)
	
	# p3 = histogram(θ(y)|>normalize,xlabel="Θ(y)")
	# p4 = histogram(Φ(x) |> normalize_mean,xlabel="Φ(x)")
	
	# p5 = Plots.scatter(x,θ(y)|>normalize,xlabel="x",ylabel="Θ(y)",leg=false,xlim=(-2.5,2.5),ylim=(-3,maximum(θ(y))))
	# p6 = Plots.scatter(y,Φ(x) |> normalize_mean,xlabel="y",ylabel="Φ(x)",leg=false)
	
	# p7 = Plots.scatter(x,Φ(x) |> normalize_mean,ylabel="Φ(x)",xlabel="x",leg=false,xlim=(-3,3))
	# p8 = Plots.scatter(y,θ(y)|>normalize,xlabel="y",ylabel="Θ(y)",leg=false,xlim=(0,3),ylim=(-1,5))
	
	# p9 = Plots.scatter(Φ(x)|> normalize_mean,θ(y)|>normalize,xlabel="Φ(x)",ylabel="θ(y)",leg=false)
	# p10 = marginalhist(Φ(x) |> normalize_mean,θ(y)|>normalize,xlabel="Φ(x)",ylabel="θ(y)",leg=false)
	
	
	# Plots.plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,margin=5Plots.px ,dpi=100,size=(900,800), layout = @layout [a b; c d; e f; g h; i j]) # 
end

begin
# ╔═╡ 026f0d85-b0d5-4076-b178-15a0371972a6
	GLMakie.activate!()
	

fontsize_theme = Theme(fontsize = 30)
set_theme!(fontsize_theme)
	
	
	set_window_config!(;
    renderloop = GLMakie.renderloop,
    vsync = true,
    framerate = 60.0,
    float = true,
    pause_rendering = false,
    focus_on_show = true,
    decorated = true,
    title = "ACE visualization"
)

	
	# xs = cos.(1:0.5:20)
	# ys = sin.(1:0.5:20)
	# zs = LinRange(0, 3, length(xs))
	nn = 0.0 .*ones(size(x))
	nn1 = 0.0 .*ones(size(z))
	# meshscatter(x,y,nn, markersize = 0.1)

	# figure, axis, plot = Makie.meshscatter(x,y,nn, markersize = 0.05, axis = (; type = Axis3, protrusions = (0, 0, 0, 0)),colormap = :inferno, transparency = true)

smoothcolor = :black
datacolor = RGBA((42, 45, 52,255)./255...)
planecolor = RGBA(0.0,0.2,1,0.05)
	
	
# Marker
sphere = Sphere(Point3f0(0,0,0), 0.5f0)
	
	
# Data on x y
figure, axs, plot = Makie.meshscatter(x,nn.-3.0, y, markersize = 0.1, label="1. Datapoints", legend=:left, axis = ( ;xlabel = "X", zlabel = "Y or Θ(y)",ylabel="Φ(x)", title = "ACE Visualization: y=exp(sin(x))", type = Axis3,aspects=:data,viewmode=:fitzoom, azimuth = 1*pi,protrusion= (0, 0, 0, 100),elevation=0.5), transparency = true,colormap = reverse(to_colormap(:viridis)), shading = true,color=datacolor,resolution=(1920,1080), marker = sphere,perspectiveness=0)
	
	
Makie.meshscatter!(x |>normalize_mean,nn.-2,(y |>normalize).+3, markersize = 0.1, axis = (; type = Axis3), transparency = true,color = RGBA((0, 157, 220,150)./255...), shading = true,alpha=0.3, marker = sphere)	

	
Makie.meshscatter!(x |>normalize_mean,nn.-1.5,y |> θ |>normalize, markersize = 0.1, axis = (; type = Axis3), transparency = true,color = RGBA((242, 100, 48,255)./255...), shading = true,alpha=0.3, marker = sphere)	

	
# # Smooth on xy
# Makie.meshscatter!(z,nn1,s, markersize = 0.1, axis = (; type = Axis3), transparency = true,colormap = reverse(to_colormap(:viridis)), shading = true,color=smoothcolor, marker = sphere)
	
# Smooth on ground
# Makie.meshscatter!(z,Φ(z)|>normalize_mean,nn1, markersize = 0.1, axis = (; type = Axis3), transparency = true,colormap = reverse(to_colormap(:viridis)), shading = true,color=smoothcolor, marker = sphere)
	
lower = [Point3f0(z[i],z[i] |> Φ, -2.0) for i in 1:length(z)]
upper = [Point3f0(z[i],z[i] |> Φ, 2.0) for i in 1:length(z)]
band!(lower, upper, axis=(type=Axis3,),transparency=true,shading=true, color=planecolor, ssao=true,alpha=0.3)	

@. zzf(z) = exp(sin(z))	
	scene = Scene(show_axis = false)

Makie.meshscatter!(x, x|>Φ |>normalize_mean, y|> θ|>normalize, markersize = 0.1, axis = (; type = Axis3), transparency = true,color = RGBA((242, 100, 48,255)./255...), shading = true,alpha=0.3, marker = sphere)	

# Scatter smooth on plane
# Makie.meshscatter!(x,s,y, markersize = 0.1, axis = (; type = Axis3), transparency = true,color = smoothcolor, shading = true, marker = sphere)	
	

# Rueckwand data
	Makie.meshscatter!(nn.+7,Φ(x)|>normalize_mean,θ(y)|>normalize, markersize = 0.1, axis = (; type = Axis3), transparency = true,color = RGBA((242, 100, 48,255)./255...), shading = true, marker = sphere)	
	

	Makie.meshscatter!(nn.+7,Φ(x)|>normalize_mean, (y|>normalize ).+3.0, markersize = 0.1, axis = (; type = Axis3), transparency = true,color =RGBA((0, 157, 220,150)./255...), shading = true, marker = sphere)	
	

	
	
# limits!(axs,-5,5,0,maximum(y),0,1.5.*maximum(y))

	# Mouse interaction
	on(events(figure).mousebutton, priority = 0) do event
    if event.button == Mouse.left
        if event.action == Mouse.press
            # println("Hallo")
        else
            # do something else when the mouse button is released
        end
    end
    # Do not consume the event
    return Consume(false)
end
	

	gl_screen = display(current_figure())
wait(gl_screen)

end
