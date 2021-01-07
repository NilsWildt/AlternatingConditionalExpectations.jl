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
