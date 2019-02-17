using Documenter, DistributionParameters

makedocs(;
    modules=[DistributionParameters],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chriselrod/DistributionParameters.jl/blob/{commit}{path}#L{line}",
    sitename="DistributionParameters.jl",
    authors="Chris Elrod",
    assets=[],
)

deploydocs(;
    repo="github.com/chriselrod/DistributionParameters.jl",
)
