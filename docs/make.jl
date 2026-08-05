# see documentation at https://juliadocs.github.io/Documenter.jl/stable/

using Documenter, AlternatingConditionalExpectations

makedocs(
    modules = [AlternatingConditionalExpectations],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "NilsWildt",
    sitename = "AlternatingConditionalExpectations.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

# Some setup is needed for documentation deployment, see “Hosting Documentation” and
# deploydocs() in the Documenter manual for more information.
deploydocs(
    repo = "github.com/NilsWildt/AlternatingConditionalExpectations.jl.git",
    push_preview = true
)
