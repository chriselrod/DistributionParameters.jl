module DistributionParameters

using PaddedMatrices, LinearAlgebra,
    SIMDPirates, SLEEFPirates, LoopVectorization, VectorizationBase,
    StackPointers, ReverseDiffExpressionsBase, SpecialFunctions#, StructuredMatrices

using PaddedMatrices: StackPointer,#, DynamicPtrMatrix,
    AbstractFixedSizeVector,
    AbstractFixedSizeMatrix,
    AbstractFixedSizeArray,
    flatvector, NoPadPtrView, AbstractStrideArray,
    memory_length_val

using VectorizationBase: One, Zero, Static, AbstractStructVec, AbstractStridedPointer
# using ReverseDiffExpressionsBase: adj
import ReverseDiffExpressionsBase: constrain
import StackPointers: stack_pointer_call

export RealFloat, RealArray, RealVector, RealMatrix, Bounds, MissingDataArray, maybe_missing,
    constrain, CorrelationMatrixCholesyFactor, AbstractParameter
#, CorrCholesky, CovarCholesky, DynamicCovarianceMatrix

abstract type AbstractParameter end

#using LoopVectorization: @vvectorize

# struct One end
# # This should be the only method I have to define.
# @inline Base.:*(::One, a) = a
# #@inline Base.:*(a, ::One) = a
# @inline Base.:*(a::StackPointer, b::One) = (a, b)
# # But I'll define this one too. Would it be better not to, so that we get errors
# # if the seed is for some reason multiplied on the right?
# #@inline Base.:*(::One, ::One) = One()
# Base.size(::One) = Core.tuple()

# @inline extract(A::LinearAlgebra.Adjoint) = A.parent
# @inline extract(A::Symmetric) = A.data
# @inline extract(A::LowerTriangular) = A.data
# @inline extract(A) = A



function constrained_length end
function parameter_names end

include("reference.jl")
include("parameter_descriptions.jl")
include("basic_constraints.jl")
include("double_bounded.jl")
include("corr_cholesky.jl")


# include("constraints.jl")
# include("uniform_mapped_parameters.jl")
# include("lkj_correlation.jl")
# include("normal_variates.jl")
# include("missing_data.jl")
# include("precompile.jl")
# _precompile_()
# include("autoregressive_matrix.jl")

# @support_stack_pointer load_parameter

# @def_stackpointer_fallback lkj_constrain ∂lkj_constrain CovarianceMatrix ∂CovarianceMatrix
# function __init__()
    # @add_stackpointer_method lkj_constrain ∂lkj_constrain CovarianceMatrix ∂CovarianceMatrix
    # _precompile_()
# end

end # module
