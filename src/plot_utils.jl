using Plots 


function scatter_bivariate(X, Y)
    return p1 = StatsPlots.scatter(
        X,
        Y,
        xlabel = "X",
        ylabel = "Y",
        m = ([:+ :h :star7], 6),
        bg = RGB(0.2, 0.2, 0.2),
        dpi = alldpi,
        cbar = false,
        leg = false,
        # xlim = (-2, 1.5),
        # ylim = (0, 60),
        view = false,
    )
    # pngsave("Comprandn(rng, Float64, (N,))arison_ACQ.png", p1)
    # imgpath = string(mkpath(plotsdir("scatters")), "/XY", ".png")
    # Plots.savefig(imgpath)
    # display(p1)
end

function mysavefig(p1, mydir, myname)
    # @warn "Here we go" mydir myname
    for ext in [".pdf",".png",".svg"]
        filename = string(projectname(), "_", myname, ext)
        plotdirname =  mydir
        try
            mkdir(joinpath(plotsdir(), plotdirname))
        catch
        end
        savefig(p1, joinpath(plotsdir(), plotdirname, filename))
    end

end


# Plot recipie.
@recipe function f(myace::ACEsim;transform=false, full=true,dpi=500,plotsize=1.5.*(700,450))
bf = myace.res
#       if length(bf.X) == 0  || !(typeof(bf.X) <: AbstractVector) ||
#         !(typeof(bf.Φ_x) <: AbstractVector)
#         error("Benchmark has wrong dimensions, or ACE solution wasn't set yet.  Got: $(typeof(bf))")
#     end

    markershape --> :circle
    markersize  --> 2
           # # set up the subplots
            link --> :none
            size-->plotsize
  xguide --> "x"
    yguide --> "y"
                    margin -->20Plots.px

        if full
            X = bf.X
            Y = bf.Y
            Φ_x = bf.Φ_x
            Θ_y = bf.Θ_y
            plot_view_bounds = bf.plot_view_bounds
            plot_fcs = bf.plot_fcs

     

            # framestyle := [:shared :shared :shared :shared]
            grid := false
            layout :=  @layout [a{0.1h}; StatsPlots.grid(2, 2)] # ;b{0.2h}
            seriestype := :scatter
                background_color := RGB(0.2, 0.2, 0.2)
                dpi:= dpi
                colorbar:= false
                legend:= false
   
                markerstrokewidth := 0
                e2 = string(round(abs((ε²(Φ_x[bf.sIx], Θ_y[bf.sIy])));digits = 4))
               @show bf
                 mytit = join(["\nACE result:\n","ε² = $e2 "])
                title:= mytit
           
   @series begin
            seriestype := :scatter
               framestyle:=:none
                subplot := 1
                end


            @series begin
                    title:=""
                     xguide := "X"
                yguide := "Y"
                    subplot := 2
                               xlims := plot_view_bounds[1][1]
                            ylims:= plot_view_bounds[1][2]
                    X,Y
                end
           
                    @series begin
                    title:=""
                         xlabel --> "X"
                ylabel --> L"\Phi(X)"
                    # xlims := plot_view_bounds[2][1]
                            ylims:= plot_view_bounds[2][2]
                    subplot := 3
                        X,Φ_x
                end

                           @series begin
                    title:=""
                xlabel --> "Y"
                ylabel --> L"\Theta(Y)"
                           xlims := plot_view_bounds[3][1]
                            ylims:= plot_view_bounds[3][2]
                    subplot := 4
                Y,Θ_y

                end

                           @series begin
                    title:=""
                xlabel --> L"\Phi(X)"
                ylabel --> L"\Theta(Y)"
                                        # xlims := plot_view_bounds[4][1]
                            ylims:= plot_view_bounds[4][2]
                    subplot := 5
                             Φ_x, Θ_y

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
