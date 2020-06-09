
struct CorrelationMatrixCholesyFactor{K}  <: AbstractParameter end

memory_length(::CorrelationMatrixCholesyFactor{K}) where {K} = (K * (K-1)) >>> 1
memory_length(::Type{<:CorrelationMatrixCholesyFactor{K}}) where {K} = (K * (K-1)) >>> 1

