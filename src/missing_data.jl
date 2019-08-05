

"""
Parameters are:
M: number Missing
T: DataType
N: 
"""
struct MissingDataArray{M,B,T,N,A <: AbstractArray{T,N}}
    data::A
    inds::MutableFixedSizePaddedVector{M,Int,M,M}
end

"""
This function is not type stable.
"""
function maybe_missing(A::AA) where {T,N,AA <: AbstractArray{Union{Missing,T},N}}
    M = sum(ismissing, A)
    M == 0 && return A
    convert(MissingDataArray{M}, A)
end

function Base.convert(::Type{<:MissingDataArray{M}}, A::AbstractArray{T}) where {M,T}
    convert(MissingDataArray{M,Bounds{typemin(T),typemax(T)}}, A)
end
function Base.convert(::Type{<:MissingDataArray{M,B}}, A::AA) where {T,B,N,AA <: AbstractArray{Union{Missing,T},N}}
#    M = sum(ismissing, A)
    #    M == 0 &&
    
    MissingDataArray{M,B,T,N,AA}(
        data, inds
    )
end


function load_parameter(
    first_pass, second_pass, out, ::Type{MissingDataArray{M,B,T}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, copyexport::Bool = false
) where {S,B,T}
    load_transformations!(
        first_pass, second_pass, B, out, Int[S.parameters...],
        partial, logjac, sptr,
        m, Symbol("##θparameter##"), Symbol("##∂θparameter##"),
        copyexport
    )    
end




