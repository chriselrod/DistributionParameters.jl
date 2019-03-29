module DistributionParameters

using PaddedMatrices, StructuredMatrices, LinearAlgebra,
        SIMDPirates, SLEEFPirates, LoopVectorization

export RealFloat, PositiveFloat, LowerBoundedFloat, UpperBoundedFloat, BoundedFloat, UnitFloat,
    RealVector, PositiveVector, LowerBoundVector, UpperBoundVector, BoundedVector, UnitVector,
    RealMatrix, PositiveMatrix, LowerBoundMatrix, UpperBoundMatrix, BoundedMatrix, UnitMatrix

"""
Transform a parameter vector to the constrained space.
"""
function constrain end

include("uniform_mapped_parameters.jl")
include("lkj_correlation.jl")
# include("autoregressive_matrix.jl")

end # module
