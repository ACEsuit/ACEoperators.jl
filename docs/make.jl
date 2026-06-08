using ACEoperators
using Documenter

DocMeta.setdocmeta!(ACEoperators, :DocTestSetup, :(using ACEoperators); recursive=true)

makedocs(;
    modules=[ACEoperators],
    authors="Christoph Ortner <christophortner0@gmail.com>",
    sitename="ACEoperators.jl",
    format=Documenter.HTML(;
        canonical="https://ACEsuit.github.io/ACEoperators.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/ACEoperators.jl",
    devbranch="main",
)
