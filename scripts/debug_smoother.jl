Base.Experimental.@optlevel 3
using DrWatson
DrWatson.quickactivate("ACE_biocat")
include("/home/nilswildt/Syncthing/Documents/Arbeit/PhD/Operativ/Projekte/ACE/Code/ACE_biocat/src/io_functions.jl")
using ACE
using DataFrames
using HDF5
using MAT
using StaticArrays
using Plots
using UUIDs
using Random
using Dates
using Distributed
using BSON
using FileIO


struct CoSorterElement{T1,T2}
    x::T1
    y::T2
end
struct CoSorter{T1,T2,S <: AbstractArray{T1},C <: AbstractArray{T2}} <: AbstractVector{CoSorterElement{T1,T2}}
    sortarray::S
    coarray::C
end

Base.size(c::CoSorter) = size(c.sortarray)
Base.getindex(c::CoSorter, i...) = 
    CoSorterElement(getindex(c.sortarray, i...), getindex(c.coarray, i...))
Base.setindex!(c::CoSorter, t::CoSorterElement, i...) = 
    (setindex!(c.sortarray, t.x, i...); setindex!(c.coarray, t.y, i...); c) 
Base.isless(a::CoSorterElement, b::CoSorterElement) = isless(a.x, b.x)
Base.Sort.defalg(v::C) where {T <: Union{Number,Missing},C <: CoSorter{T}} = 
    Base.DEFAULT_UNSTABLE

function sort_two_arrays(x::Array, y::Array)
    T = CoSorter(x, y)
    sort!(T)
    x = T.sortarray
    y = T.coarray
    return x,y
end



    response  = :Ka # Find out, which ones I can "predict" :Kp,:Ka, :Kb, :Kia,:Keq,:Kins,:Kcatf
    predictor= [:Kb, :Kia] # [:Ka, :Kb, :Kia],[:Kb, :Kia],[:Kb]
    numSamples  = 5.0# in percent
    scale_data = false #true  # See the impact of scaling
    # windowsize = 128 # @onlyif(:smoother != ACE.FRSS , 8),@onlyif(:smoother != ACE.FRSS , 10),@onlyif(:smoother != ACE.FRSS , 16),@onlyif(:smoother != ACE.FRSS , 32)]

    df = read_hd5_chain(get_hd5_files()[1])
      chain = Array{Float64}(df)
      howmany = Int64(ceil(numSamples.*length(chain)./100))
      fac = Int64(ceil(length(chain)÷howmany))
      j = findfirst(x -> x == string(response), string.(names(df)))
      Y = Array{Float64,2}(chain[1:fac:end,j][:,:])

        X = zeros(length(Y),length(predictor))
        for (k,p) in enumerate(predictor)
            if p!=response
                i =findfirst(x -> x == string(p), string.(names(df)))
                @show i 
                X[:,k] = Array{Float64, 2}(chain[1:fac:end,i][:,:])
            end
        end

        if scale_data == true
            Y = Y./maximum(Y)
            for j in size(X,2)
                X[:,j] =  X[:,j]./maximum( X[:,j])
            end
            @info "Scaled all data."
        end



        @show size(X) typeof(X)

        Nk = length(X)

        if isodd(Nk)
            @warn "Hey, Nk is Odd"
        else
                    @info "Hey, Nk is even"
        end
        # mysmoother1 = ACE.LAS(Nk ÷ 2)
        # mysmoother2 = ACE.LASb(Nk ÷ 5)
        # mysmoother3 =  ACE.LLSS(Nk ÷ 8)
        # mysmoother4 =  ACE.LLSSb(Nk ÷ 6)
        # mysmoother5 = ACE.FRSS([0.05,0.1,0.5], 0.2, 0.2)
        # σ = 0.1
        # mykernel = ACE.Kernelregression.Gaussian(σ)
        # reg = 1.0
        # mysmoother2 = ACE.NWKernelsmooth(mykernel)
        # mysmoother = undef
        # @show smoother typeof(smoother)
        #         if smoother != ACE.FRSS
        #         mysmoother = smoother(Nk÷windowsize)
        #         elseif smoother == ACE.FRSS
        #         mysmoother = smoother([0.05,0.2,0.5], 0.2, 0.2)
        # end
        xx = Vector(X[:,2])
        yy = Vector(Y[:])
        x,y = sort_two_arrays(xx,yy)
        mysmoother = ACE.LAS(40)
        # mysmoother = ACE.FRSS([0.05,0.1,0.5], 0.2, 0.2)

        # Create smoother stack, for multiple smoother
      smoothY =   do_smoothing(xx,yy,mysmoother)
      h1 = scatter(xx,yy)
        if length(filter(isnan,smoothY))>0
        @warn "There are NaNs!"
      end
      plot!(h1,xx,smoothY)


    scatter(x,y,markersize=0.1)