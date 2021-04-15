using Random
using DocStringExtensions
using Documenter
using Plots
using StatsPlots
using LaTeXStrings
using Latexify
using DrWatson
include(srcdir("utils.jl"))
abstract type BenchmarkFunction end


# Plot recipie.
@recipe function f(bf::BenchmarkFunction;transform=false, full=true)
#       if length(bf.X) == 0  || !(typeof(bf.X) <: AbstractVector) ||
#         !(typeof(bf.Φ_x) <: AbstractVector)
#         error("Benchmark has wrong dimensions, or ACE solution wasn't set yet.  Got: $(typeof(bf))")
#     end

    markershape --> :circle
    markersize  --> 2


        if full
            X = bf.X
            Y = bf.Y
            Φ_x = bf.Φ_x
            Θ_y = bf.Θ_y
            # plot_view_bounds = bf.plot_view_bounds
            # plot_fcs = bf.plot_fcs

            # # set up the subplots
            legend := false
            link := :both
            # framestyle := [:none :axes :none]
            grid := false
            layout := 4
             seriestype := :scatter
       
                

                @series begin
                    subplot := 1
                    #            xlims := plot_view_bounds[1][1]
                    #         ylims:= plot_view_bounds[1][2]
                    X,Y
                end

                    @series begin
                    subplot := 2
                    #    xlims := plot_view_bounds[1][1]
                    # ylims:= plot_view_bounds[1][2]
                    X,Y
                end

                           @series begin
                    subplot := 3
                    #    xlims := plot_view_bounds[1][1]
                    # ylims:= plot_view_bounds[1][2]
                    X,Y
                end

                           @series begin
                    subplot := 4
                    #    xlims := plot_view_bounds[1][1]
                    # ylims:= plot_view_bounds[1][2]
                    X,Y
                end

                       
        else 

        if transform && length(bf.Φ_x ) != 0 
            x:= bf.Φ_x
            y:= bf.Θ_y
        else
        x:=bf.X
        y:=bf.Y
        end
    end
    # ()
end


"""
4 Plots for bivariate ACE benchmarks.
===


Use: 
$(TYPEDSIGNATURES)
"""
function benchmark_ace_plot(bf::BenchmarkFunction, vargs...)
    X = bf.X
    Y = bf.Y
    Φ_x = bf.Φ_x
    Θ_y = bf.Θ_y

    plot_view_bounds = bf.plot_view_bounds
    plot_fcs = bf.plot_fcs


    alldpi = 300
    if length(vargs) >= 1
        alldpi = vargs[2]
    end
    p1 = StatsPlots.scatter(
                X,
                Y,
                xlabel = "X",
                ylabel = "Y",
                m = ([:circle], 2),
                bg = RGB(0.2, 0.2, 0.2),
                dpi = alldpi,
                cbar = false,
                leg = false,
                xlim = plot_view_bounds[1][1],
                ylim = plot_view_bounds[1][2],
                markerstrokewidth = 0
            )
    p2 = StatsPlots.scatter(
               X,Φ_x,
                xlabel = "X",
                ylabel = L"\Phi(X)",
                m = ([:circle], 2),
                bg = RGB(0.2, 0.2, 0.2),
                dpi = alldpi,
                cbar = false,
                leg = false,
                xlim = plot_view_bounds[2][1],
                ylim = plot_view_bounds[2][2],
                markerstrokewidth = 0
            )
   if length(plot_fcs) >0
    for (n,f) in  plot_fcs[1]
        @debug "Plot 1,2, X vs X" f
        flabel = ""
        try
            try
         flabel = latexify(n)
            catch
                flabel = convert(String,n)
            end
        catch
            flabel = "undef"
        end
       p2 =  Plots. plot!(p2, sort(X;dims = 1),f( sort(X;dims = 1)), label = flabel, legend = :best)
    end
end
    # X, Y = sort_two_arrays(X, Y)
    # p3Y, p3Θ_y = sort_two_arrays(Y, Θ_y)
    p3 = StatsPlots.scatter(
                Y,Θ_y,
                xlabel = "Y",
                ylabel = L"\Theta(Y)",
                m = ([:circle], 2),
                bg = RGB(0.2, 0.2, 0.2),
                dpi = alldpi,
                cbar = false,
                leg = false,
                  xlim = plot_view_bounds[3][1],
                ylim = plot_view_bounds[3][2],
                markerstrokewidth = 0
            )
if length(plot_fcs) >1 
for (n,f) in  plot_fcs[2]
       flabel = ""
       try
            try
         flabel = latexify(n)
            catch
                flabel = convert(String,n)
            end
        catch
            flabel = "undef"
        end
          Plots.plot!(p3, sort(Y;dims = 1), abs.(f.(sort(Y;dims = 1))).^(1 / 3), label = flabel, legend = :best)
    end
end
    # plot!(p3, sort(Y), (sort(Y).^(1 / 3)))

    # X, Φ_x
    p4 = StatsPlots.scatter(
                Φ_x, Θ_y,
                xlabel = L"\Phi(X)",
                ylabel = L"\Theta(Y)",
                m = ([:circle], 2),
                bg = RGB(0.2, 0.2, 0.2),
                dpi = alldpi,
                cbar = false,
                leg = false,
                         xlim = plot_view_bounds[4][1],
                ylim = plot_view_bounds[4][2],
                markerstrokewidth = 0
            )
        p5 = Plots.scatter(1:length(Benchmark.conv_err), Benchmark.conv_err)
        
        # scatter!(X.^3, log.(Y), m = (:dot, 1))
    if length(vargs) >= 1
        e2 = string(round(abs((ε²(Φ_x[Benchmark.sIx], Θ_y[Benchmark.sIy])));digits = 4))
        mytit = join([vargs[1] , " ε² = $e2 "])
        l = @layout [a{0.03h}; StatsPlots.grid(2, 2);b{0.2h}]
        title = Plots.plot(title = mytit, grid = false, showaxis = false, bottom_margin = -50Plots.px) 
       StatsPlots. plot(title, p1, p2, p3, p4,p5, layout = l,  size = 0.8 .* (1.6 * 1000, 1000))
    else
        l = @layout [a b; c d]
        plot(p1, p2, p3, p4, layout = l)
    end
end

"""
Heaviside
===

Calculate the heaviside function.

Use: 
$(TYPEDSIGNATURES)
"""
function heaviside(x)
    return @.  0.5 * (sign(x) + 1.0)
end


"""
Normal sampler 
===

See above.

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector


σ: std and μ: mean 

use_seed: true or false

seed: 
"""
function normal_sample(numSamples::Int64, numDim::Int64, μ::Float64 , σ::Float64 , use_seed::Bool = false, seed::Int64 = 0)
    dims = (numSamples, numDim)
    rng =     MersenneTwister() 
  if use_seed
        rng =  MersenneTwister(seed)    
    end
    return  σ .* randn(rng,  Float64, dims) .+ μ
end



"""
Uniform sampler 
===

See above.

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

bounds: Tuple(lower_bound, upper_bound)

use_seed: true or false

seed: 
"""
function uniform_sample(numSamples::Int64, numDim::Int64,  σ_noise::Float64, use_seed::Bool, seed::Int64 = 0, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
    lb, ub = bounds
    dims = (numSamples, numDim)
       rng =     MersenneTwister() 
  if use_seed
        rng =  MersenneTwister(seed)    
    end
    return abs(ub - lb) .* (rand(rng,  Float64, dims)) .+ lb
end

"""
$(FUNCTIONNAME)
===

Returns the correct sample, dependent on "samplingmethod"

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

samplingmethod: 1. "normal" 2. "uniform"

use_seed: true or false

seed: 

bounds
"""
function get_sample(numSamples::Int64, numDim::Int64, samplingmethod::String, σ::Float64, use_seed::Bool, seed::Int64 = 0, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
         if samplingmethod == "uniform"
        X = uniform_sample(numSamples, numDim, σ, use_seed, seed, bounds)
    elseif samplingmethod == "normal"
        X = normal_sample(numSamples, numDim, 0.0, σ, use_seed, seed)
    else 
        @warn "You didn't provide a sampling method. Using normal distributed samples now."
           X = normal_sample(numSamples, numDim,0.0,1.0, use_seed, seed)
    end
    return X
end

"""
$(FUNCTIONNAME)
===

$(raw"""
f_{b1}(x_1)= x_2 = \exp{(x_1^3 + \varepsilon)})
""")

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

samplingmethod: 1. "normal" 2. "uniform"

use_seed: true or false

seed: 

bounds
"""
mutable struct f_b1 <: BenchmarkFunction
numSamples::Int64
numDim::Int64
samplingmethod::String
σ_x::Float64
σ_noise::Float64
use_seed::Bool
seed::Int64
scale_data::Bool
bounds::Tuple{Float64,Float64}
X::AbstractArray
Y::AbstractArray
# ρ::Float64 # Correlation
# e²::Float64
plot_view_bounds::AbstractArray
plot_fcs::AbstractArray
name::String
scale_factors::AbstractArray
Φ_x::AbstractArray
Θ_y::AbstractArray
sIx::AbstractArray
sIy::AbstractArray
bsIx::AbstractArray
bsIy::AbstractArray
conv_err::AbstractArray


function f_b1(numSamples::Int64, numDim::Int64, samplingmethod::String, σ_x::Float64, σ_noise::Float64, use_seed::Bool, seed::Int64,scale_data::Bool, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
lb, ub = bounds
X = get_sample(numSamples, numDim, samplingmethod, σ_x, use_seed, seed, bounds)
Y =  exp.(X.^3 + normal_sample(numSamples, numDim, 0.0, σ_noise, use_seed, seed + 1))
viewbounds = [[(-2.5, 2.5),(0, 80)], [(-2.5, 2.5),(-2.5, 2.5)],[(0, 80),(-2, 5)],[(-3, 3),(-3, 3)]]
 scale_factors  = [ zeros(2), zeros(2)]
if scale_data == true
    xmax = maximum(X) 
    xmin = minimum(X)
    ymax = maximum(Y)  
    ymin = minimum(Y)
    scale_factors=[[xmin,xmax],[ymin,ymax]]
    X = normalize.(X,xmin,xmax)
    Y= normalize.(Y,ymin,ymax)
viewbounds = [[(0,1),(0,1)],[(0,1),(0,1)],[(0,1.0),(0,1.5)],[(-3, 3),(-3, 3)]]
end
plot_fcs =  [Dict(L"x^3"=> x -> x.^3,L"x"=> x ->x), Dict(L"\log(x)"=> x -> log.(x),L"3throot\log(x)"=> x -> log.(x.^(1/3)))]
new(numSamples, numDim,samplingmethod, σ_x,σ_noise, use_seed,seed, scale_data,bounds,X,Y, viewbounds,plot_fcs,L"BenchmarkFunction 1: $f_{b1}(x_1)= x_2 = \exp{(x_1^3 + \varepsilon)}$ ",scale_factors) 
end
end



function normalize(X::Float64,a::Float64,b::Float64)
return  (X-a)/(b-a)
end

# function denormalize!(fb::BenchmarkFunction)
# a = fb.scale_factors[1][1]
# b = fb.scale_factors[1][2]
# c = fb.scale_factors[2][1]
# d = fb.scale_factors[2][2]
# fb.Φ_x = fb.Φ_x.*(b.-a).+a
# fb.Φ_x = fb.Φ_x.*(b.-a).+a
# end


"""
$(FUNCTIONNAME)
===

$(raw"""
f_{b2}(x_1)= x_2 = \exp{(\sin{(x_1)} + \varepsilon)}
""")

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

samplingmethod: 1. "normal" 2. "uniform"

use_seed: true or false

seed: 

bounds
"""
mutable struct f_b2 <: BenchmarkFunction
numSamples::Int64
numDim::Int64
samplingmethod::String
σ_x::Float64
σ_noise::Float64
use_seed::Bool
seed::Int64
scale_data::Bool
bounds::Tuple{Float64,Float64}
X::AbstractArray
Y::AbstractArray
# ρ::Float64 # Correlation
# e²::Float64
plot_view_bounds::AbstractArray
plot_fcs::AbstractArray
name::String
scale_factors::AbstractArray
Φ_x::AbstractArray
Θ_y::AbstractArray
sIx::AbstractArray
sIy::AbstractArray
bsIx::AbstractArray
bsIy::AbstractArray
conv_err::AbstractArray

function f_b2(numSamples::Int64, numDim::Int64, samplingmethod::String, σ_x::Float64, σ_noise::Float64, use_seed::Bool, seed::Int64,scale_data::Bool, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
     lb, ub = bounds
     X = get_sample(numSamples, numDim, samplingmethod, σ_x,use_seed, seed, bounds)
    Y =  exp.(sin.(X) +  normal_sample(numSamples, numDim,0.0,σ_noise,use_seed, seed + 1))
    viewbounds = [[(lb, ub),(0, 40)], [(lb,ub),(-ub,ub)],[(0, 80),(-2, 5)],[(-3, 3),(-3, 3)]]   
    scale_factors  = [ zeros(2), zeros(2)]
if scale_data == true
    xmax = maximum(X) 
    xmin = minimum(X)
    ymax = maximum(Y) 
    ymin = minimum(Y)
    scale_factors=[[xmin,xmax],[ymin,ymax]]
    X = normalize.(X,xmin,xmax)
    Y= normalize.(Y,ymin,ymax)
viewbounds = [[(0, 1),(0, 1)], [(-2.0,2.0),(-2.0,2.0)],[(-2.0,2.0),(-2.0,2.0)],[(-3, 3),(-3, 3)]]
end
plot_fcs =  [Dict("sin(x)"=>x -> sin.(x)), Dict("log(x)"=> x -> log.(x))]
new(numSamples, numDim,samplingmethod, σ_x,σ_noise, use_seed, seed,scale_data, bounds,X,Y, viewbounds,plot_fcs,L"BenchmarkFunction 2: $f_{b2}(x_1)= x_2 = \exp{(\sin{(x_1)} + \varepsilon)}$ ",scale_factors) 
end
end



"""
$(FUNCTIONNAME)
===

$(raw"""
f_{b3}(x_1)= x_2 = \sigma(x_2)
""")

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

samplingmethod: 1. "normal" 2. "uniform"

use_seed: true or false

seed: 

bounds
"""
mutable struct f_b3 <: BenchmarkFunction
numSamples::Int64
numDim::Int64
samplingmethod::String
σ_x::Float64
σ_noise::Float64
use_seed::Bool
seed::Int64
scale_data::Bool
bounds::Tuple{Float64,Float64}
X::AbstractArray
Y::AbstractArray
# ρ::Float64 # Correlation
# e²::Float64
plot_view_bounds::AbstractArray
plot_fcs::AbstractArray
name::String
scale_factors::AbstractArray
Φ_x::AbstractArray
Θ_y::AbstractArray
sIx::AbstractArray
sIy::AbstractArray
bsIx::AbstractArray
bsIy::AbstractArray
conv_err::AbstractArray

function f_b3(numSamples::Int64, numDim::Int64, samplingmethod::String, σ_x::Float64, σ_noise::Float64, use_seed::Bool, seed::Int64,scale_data::Bool, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
        lb, ub = bounds
     X = get_sample(numSamples, numDim, samplingmethod, σ_x,use_seed, seed, bounds)
    Y = 2. .*heaviside.(X) .+ normal_sample(numSamples, numDim,0.0,σ_noise,use_seed, seed + 1)
    viewbounds = [[(lb, ub),(-1.3, 5)], [(lb,ub),(lb,ub)],[(-5, 5),(-5, 5)],[(-3, 3),(-3, 3)]]

   scale_factors  = [ zeros(2), zeros(2)]
if scale_data == true
    xmax = maximum(X) 
    xmin = minimum(X)
    ymax = maximum(Y) 
    ymin = minimum(Y)
    scale_factors=[[xmin,xmax],[ymin,ymax]]
    X = normalize.(X,xmin,xmax)
    Y= normalize.(Y,ymin,ymax)
viewbounds = [[(0, 1),(0, 1)], [(0,1),(0,1)],[(0, 1),(0,1)],[(-3, 3),(-3, 3)]]
end
plot_fcs =  [] # Dict("σ(x)"=>x -> heaviside.(x))
new(numSamples, numDim,samplingmethod, σ_x,σ_noise, use_seed, seed,scale_data, bounds,X,Y, viewbounds,plot_fcs,L"BenchmarkFunction 3: $f_{b3}(x_1)= x_2 = \sigma(x_2)$ ",scale_factors) 
end
end





"""
$(FUNCTIONNAME)
===

$(raw"""
f_{b4}(x_1)= x_2 = \log{(\log{(x_2)})}
""")

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

samplingmethod: 1. "normal" 2. "uniform"

use_seed: true or false

seed: 

bounds
"""
mutable struct f_b4 <: BenchmarkFunction
numSamples::Int64
numDim::Int64
samplingmethod::String
σ_x::Float64
σ_noise::Float64
use_seed::Bool
seed::Int64
scale_data::Bool
bounds::Tuple{Float64,Float64}
X::AbstractArray
Y::AbstractArray
# ρ::Float64 # Correlation
# e²::Float64
plot_view_bounds::AbstractArray
plot_fcs::AbstractArray
name::String
scale_factors::AbstractArray
Φ_x::AbstractArray
Θ_y::AbstractArray
sIx::AbstractArray
sIy::AbstractArray
bsIx::AbstractArray
bsIy::AbstractArray
conv_err::AbstractArray

function f_b4(numSamples::Int64, numDim::Int64, samplingmethod::String, σ_x::Float64, σ_noise::Float64, use_seed::Bool, seed::Int64, scale_data::Bool,bounds::Tuple{Float64,Float64} = (0.0, 1.0))
     lb, ub = bounds
     X = get_sample(numSamples, numDim, samplingmethod, σ_x,use_seed, seed, bounds)
    Y = log.(abs.(X +normal_sample(numSamples, numDim,0.0,σ_noise,use_seed, seed + 1)))
viewbounds = [[(lb, ub),(-5, 5)], 
                                [(lb,ub),(-5,ub)],
                                [(lb, ub),(-2, 5)],
                                [(-2.5, 2.5),
                                (-2.5,2.5)]
                                ]


   scale_factors  = [ zeros(2), zeros(2)]
if scale_data == true
    xmax = maximum(X) 
    xmin = minimum(X)
    ymax = maximum(Y) 
    ymin = minimum(Y)
    scale_factors=[[xmin,xmax],[ymin,ymax]]
    X = normalize.(X,xmin,xmax)
    Y= normalize.(Y,ymin,ymax)
viewbounds = [[(0, 1),(0, 1)], [(0,1),(0,1)],[(0, 1),(0,1)],[(-3, 3),(-3, 3)]]
end
plot_fcs =  [Dict("abs(x)"=>x -> abs.(x)), Dict("exp(x)"=> x -> exp.(x))]
new(numSamples, numDim,samplingmethod, σ_x,σ_noise, use_seed, seed,scale_data, bounds,X,Y, viewbounds,plot_fcs,L"BenchmarkFunction 2: $f_{b2}(x_1)= x_2 = \exp{(\sin{(x_1)} + \varepsilon)}$ ",scale_factors) 
end
end



"""
$(FUNCTIONNAME)
===

$(raw"""
f_{toy} = X + noise
""")

Use: 
$(TYPEDSIGNATURES)

with

numSamples: Number of sampels (length of the vector)

numDim: Number of rows of the vector

samplingmethod: 1. "normal" 2. "uniform"

use_seed: true or false

seed: 

bounds
"""
mutable struct f_toy <: BenchmarkFunction
numSamples::Int64
numDim::Int64
samplingmethod::String
σ_x::Float64
σ_noise::Float64
use_seed::Bool
seed::Int64
scale_data::Bool
bounds::Tuple{Float64,Float64}
X::AbstractArray
Y::AbstractArray
# ρ::Float64 # Correlation
# e²::Float64
plot_view_bounds::AbstractArray
plot_fcs::AbstractArray
name::String
scale_factors::AbstractArray
Φ_x::AbstractArray
Θ_y::AbstractArray
sIx::AbstractArray
sIy::AbstractArray
bsIx::AbstractArray
bsIy::AbstractArray
conv_err::AbstractArray
function f_toy(numSamples::Int64, numDim::Int64, samplingmethod::String, σ_x::Float64, σ_noise::Float64, use_seed::Bool, seed::Int64,scale_data::Bool, bounds::Tuple{Float64,Float64} = (0.0, 1.0))
     lb, ub = bounds
     X = get_sample(numSamples, numDim, samplingmethod, σ_x,use_seed, seed, bounds)
    Y = X +normal_sample(numSamples, numDim,0.0,σ_noise,use_seed, seed + 1)
viewbounds = [[(lb, ub),(-5, 5)], 
                                [(lb,ub),(-5,ub)],
                                [(lb, ub),(-2, 5)],
                                [(-2.5, 2.5),
                                (-2.5,2.5)]
                                ]

   scale_factors  = [ zeros(2), zeros(2)]
if scale_data == true
    xmax = maximum(X) 
    xmin = minimum(X)
    ymax = maximum(Y) 
    ymin = minimum(Y)
    scale_factors=[[xmin,xmax],[ymin,ymax]]
    X = normalize.(X,xmin,xmax)
    Y= normalize.(Y,ymin,ymax)
viewbounds = [[(0, 1),(0, 1)], [(0,1),(0,1)],[(0, 1),(0,1)],[(-3, 3),(-3, 3)]]
end
plot_fcs =  [Dict("x"=>x -> x), Dict("x"=> x -> x)]
new(numSamples, numDim,samplingmethod, σ_x,σ_noise, use_seed, seed, scale_data,bounds,X,Y, viewbounds,plot_fcs,L"BenchmarkFunction 2: $f_{b2}(x_1)= x_2 = \exp{(\sin{(x_1)} + \varepsilon)}$ ",scale_factors) 
end
end



function Base.show(io::IO, bf::BenchmarkFunction)
    if !(isdefined(Main, :IJulia) && Main.IJulia.inited)
    show(io, MIME("text/plain"), bf.name)
else
     print(io, bf.name)
    end
end


 function Base.show(io::IO, ::MIME"text/html",bf::BenchmarkFunction) 
     print(io,bf.name)
end



"""
from https://github.com/korsbo/Latexify.jl/blob/master/src/utils.jl#L19
    render(::LaTeXString; debug=false, name=tempname(), command="\\Large")
Display a standalone PDF with the given input.
"""
function render(s::LaTeXString; debug=false, name=tempname(), command="\\Large")
    doc = """
    \\documentclass[varwidth=100cm]{standalone}
    \\usepackage{amssymb}
    \\usepackage{amsmath}
    $(occursin("\\ce{", s) ? "\\usepackage{mhchem}" : "")
    \\begin{document}
    {
        $command
        $s
    }
    \\end{document}
    """
    doc = replace(doc, "\\begin{align}"=>"\\[\n\\begin{aligned}")
    doc = replace(doc, "\\end{align}"=>"\\end{aligned}\n\\]")
    doc = replace(doc, "\\require{mhchem}\n"=>"")
    open("$(name).tex", "w") do f
        write(f, doc)
    end
    cd(dirname(name)) do 
        cmd = `lualatex --interaction=batchmode $(name).tex`
        debug || (cmd = pipeline(cmd, devnull))
        run(cmd)
    end
    if Sys.iswindows()
        run(`cmd /c "start $(name).pdf"`, wait=false)
    elseif Sys.islinux()
        run(`xdg-open $(name).pdf`, wait=false)
    elseif Sys.isapple()
        run(`open $(name).pdf`, wait=false)
    elseif Sys.isbsd()
        run(`xdg-open $(name).pdf`, wait=false)
    end
    return nothing
end
