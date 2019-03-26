module DistributionParameters

using PaddedMatrices, StructuredMatrices, LinearAlgebra,
        SIMDPirates, SLEEFPirates, LoopVectorization

export RealFloat, PositiveFloat, LowerBoundedFloat, UpperBoundedFloat, BoundedFloat, UnitFloat,
    RealVector, PositiveVector, LowerBoundVector, UpperBoundVector, BoundedVector, UnitVector,
    RealMatrix, PositiveMatrix, LowerBoundMatrix, UpperBoundMatrix, BoundedMatrix, UnitMatrix

include("uniform_mapped_parameters.jl")
include("lkj_correlation.jl")
# include("autoregressive_matrix.jl")

end # module
