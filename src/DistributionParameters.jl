module DistributionParameters

using PaddedMatrices, StructuredMatrices, LinearAlgebra,
        SIMDPirates, SLEEFPirates, LoopVectorization, VectorizationBase

export RealFloat, PositiveFloat, LowerBoundedFloat, UpperBoundedFloat, BoundedFloat, UnitFloat,
    RealVector, PositiveVector, LowerBoundVector, UpperBoundVector, BoundedVector, UnitVector,
    RealMatrix, PositiveMatrix, LowerBoundMatrix, UpperBoundMatrix, BoundedMatrix, UnitMatrix,
    constrain, MultivariateNormalVariate, CovarianceMatrix

"""
Transform a parameter vector to the constrained space.
"""
function constrain end

include("uniform_mapped_parameters.jl")
include("lkj_correlation.jl")
include("normal_variates.jl")
# include("autoregressive_matrix.jl")

end # module
