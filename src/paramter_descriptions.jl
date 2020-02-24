

# struct Bounds{L,U} end
# Bounds() = Bounds{-Inf,Inf}()
# Base.min(::Bounds{L}) where {L} = L
# Base.max(::Bounds{L,U}) where {L,U} = U

struct RealScalar{L,U}
    RealScalar{L,U}() where {L,U} = new{Float64(L),Float64(U)}()
end
struct RealArray{S<:Tuple,L,U,D}
    dynamicsizes::NTuple{D,Int}
    RealArray{S,L,U,D}(d::NTuple{D,Int}) where {S,L,U,D} = ((@assert U > L); new{S,Float64(L),Float64(U),D}(d))
end
const RealVector{N,L,U,D} = RealArray{Tuple{N},L,U,D}
const RealMatrix{M,N,L,U,D} = RealArray{Tuple{M,N},L,U,D}
RealScalar() = RealScalar{-Inf,Inf}()
RealScalar{L}() where {L} = RealScalar{Float64(L),Inf}()
RealArray{S}() where {S} = RealArray{S,-Inf,Inf,0}(tuple())
RealArray{S,L}() where {S,L} = RealArray{S,Float64(L),Inf,0}(tuple())
RealArray{S,L,U}() where {S,L,U} = RealArray{S,Float64(L),Float64(U),0}(tuple())
RealArray{S,L,U,0}() where {S,L,U} = RealArray{S,Float64(L),Float64(U),0}(tuple())
RealArray{S}(ds::NTuple{D}) where {S<:Tuple,D} = RealArray{S,-Inf,Inf,D}(ds)
RealArray{S,L}(ds::NTuple{D}) where {S<:Tuple,L,D} = RealArray{S,Float64(L),Inf,D}(ds)
RealArray{S,L,U}(ds::NTuple{D}) where {S<:Tuple,L,U,D} = RealArray{S,Float64(L),Float64(U),D}(ds)
RealArray(ds::NTuple{1,Int}) = RealArray{Tuple{-1},-Inf,Inf,1}(ds)
RealArray(ds::NTuple{2,Int}) = RealArray{Tuple{-1,-1},-Inf,Inf,2}(ds)
RealArray{L}(ds::NTuple{1,Int}) where {L} = RealArray{Tuple{-1},Float64(L),Inf,1}(ds)
RealArray{L}(ds::NTuple{2,Int}) where {L} = RealArray{Tuple{-1,-1},Float64(L),Inf,2}(ds)
RealArray{L,U}(ds::NTuple{1,Int}) where {L,U} = RealArray{Tuple{-1},Float64(L),Float64(U),1}(ds)
RealArray{L,U}(ds::NTuple{2,Int}) where {L,U} = RealArray{Tuple{-1,-1},Float64(L),Float64(U),2}(ds)

using PaddedMatrices
import PaddedMatrices: param_type_length
param_type_length(::RealScalar) = 1
param_type_length(::RealVector{M}) where {M} = M
param_type_length(::RealMatrix{M,N}) where {M,N} = M*N
@generated PaddedMatrices.param_type_length(::RealArray{S}) where {S} = PaddedMatrices.simple_vec_prod(S.parameters)



abstract type AbstractMissingDataArray{M,L,U,T,N,A <: AbstractArray{T,N}} end
param_type_length(::AbstractMissingDataArray{M}) where {M} = M
"""
Parameters are:
M: number Missing
B: Bounds
T: DataType
N: Dimensionality of Array (how many axis?)
"""
struct MissingDataArray{M,L,U,T,N,A} <: AbstractMissingDataArray{M,L,U,T,N,A}
    data::A
    inds::Vector{CartesianIndex{N}}
end
struct ThreadedMissingDataArray{M,L,U,T,N,A} <: AbstractMissingDataArray{M,L,U,T,N,A}
    data::Vector{A}
    inds::Vector{CartesianIndex{N}}
end


@enum ParameterType::Int8 begin
    RealScalarParameter
    RealArrayParameter
    LKJCorrCholeskyParameter
    SimplexParameter
end


"""
The basic plan is to use Bounds{L,U} for dispatch.
Generated functions will switch to BoundsValues internally, so that the code
generated the function bodies will not have to recompile.
"""
struct Bounds
    lb::Float64
    ub::Float64
end

Bounds(::RealScalar{L,U}) where {L,U} = Bounds(L,U)
Bounds(::RealArray{S,L,U}) where {S,L,U} = Bounds(L,U)

Base.min(b::Bounds) = b.lb
Base.max(b::Bounds) = b.ub
ismin(x::T) where {T} = ((x == typemin(T)) | (x == -floatmax(T))) # floatmin
ismax(x::T) where {T} = ((x == typemax(T)) | (x ==  floatmax(T)))
function isunbounded(b::BoundsValue)
    ismin(b.lb) & ismax(b.ub)
end
function islowerbounded(b::BoundsValue)
    (!ismin(b.lb)) & ismax(b.ub)
end
function isupperbounded(b::BoundsValue)
    ismin(b.lb) & (!ismax(b.ub))
end
function isbounded(b::BoundsValue)
    (!ismin(b.lb)) & (!ismax(b.ub))
end


# struct ParamDescriptionCore
#     b::Bounds
#     ind::Int32
#     t::ParameterType
# end

# struct SizedParamDescription{N}
#     s::NTuple{N,Int}
#     d::ParamDescriptionCore
# end
struct LengthParamDescription
    l::Int
    ind::Int32
    r::Int8
    LengthParamDescription(l::Int, d) = new(l, d % Int32, (l & (VectorizationBase.REGISTER_SIZE>>>3) ) % Int8)
end
LengthDescription(A, ind) = LengthDescription(param_type_length(A)::Int, ind % Int32)

# LengthParamDescription(spd::SizedParamDescription{0}) = LengthParamDescription(1, spd.d)#, one(Int8))
# function LengthParamDescription(spd::SizedParamDescription)
#     s = spd.s
#     l = minimum(s) < 0 ? -1 : prod(s)
#     LengthParamDescription(l, spd.d)
# end
# function update_rem!(descript::AbstractVector{LengthParamDescription}, R)
#     Rm1 = R - 1
#     for j ∈ 1:length(descript)
#         d = descript[j]
#         d = LengthParamDescription(d.l, d.ind, (d.l & Rm1) % Int8)
#         descript[j] = d
#     end
# end

function Base.isless(lpd1::LengthParamDescription, lpd2::LengthParamDescription)
    l1 = lpd1.l; l2 = lpd2.l
    l1dR = iszero(lpd1.r)
    l2dR = iszero(lpd2.r)
    if l1dR ⊻ l2dR
        l1dR
    else#if l1dR & l2dR
        l1 > l2
    # else
        # isless(lpd1, lpd2, R >>> 1)
    end
end
function shift_entries_right!(x, by = 1, start = firstindex(x), stop = lastindex(x))
    @assert stop ≤ lastindex(x)
    @assert start ≥ firstindex(x)
    @inbounds for i ∈ stop-by:-1:start
        x[i+by] = x[i]
    end
end
# Search for parameter pairs that preserve alignemnt, and move them to the front of unaligned params.
# E.g., given [5,5,5,3,3,1] and 8-Float64 alignment, move the (5,3)s in front, reordering to [5,3,5,3,5,1]
function combine_param_pairs!(descript::AbstractVector{LengthParamDescription}, R, start = firstindex(descript), jstart = start)
    N = length(descript)
    start > N - 2 && return start
    for j ∈ jstart:N
        descriptⱼ = descript[j]
        rⱼ = descriptⱼ.r
        Rdiff = R - rⱼ
        for k ∈ j+1:N
            descriptₖ = descript[k]
            rₖ = descriptₖ.r
            if rₖ == Rdiff
                shift_entries_right!(descript, 1, j + 1, k)
                shift_entries_right!(descript, 2, start, j + 1)
                descript[start] = descriptⱼ
                descript[start+1] = descriptₖ
                return combine_param_pairs!(descript, R, start + 2, j + 2)
            end
        end
    end
    start
end
# Search for parameter tripples that preserve alignment, and move them in front of unaligned params.
# E.g., given [5,5,5,2,2,1] and 8-Float64 alignment, move (5,2,1) in front so that it is [5,2,1,5,5,2]
function combine_param_tripples!(descript::AbstractVector{LengthParamDescription}, R, start = firstindex(descript), istart = start)
    N = length(descript)
    start > N - 3 && return start
    for i ∈ start:N
        descriptᵢ = descript[i]
        rᵢ = descriptᵢ.r
        Rdiffᵢ = R - rᵢ
        for j ∈ i+1:N
            descriptⱼ = descript[j]
            rⱼ = descriptⱼ.r
            Rdiffⱼ = Rdiffᵢ - rⱼ
            for k ∈ j+1:N
                descriptₖ = descript[k]
                rₖ = descriptₖ.r
                if rₖ == Rdiffⱼ
                    shift_entries_right!(descript, 1, j + 1, k)
                    shift_entries_right!(descript, 2, i + 1, j + 1)
                    shift_entries_right!(descript, 3, start, i + 1)
                    descript[start  ] = descriptᵢ
                    descript[start+1] = descriptⱼ
                    descript[start+2] = descriptₖ
                    return combine_param_tripples!(descript, R, start + 3, i + 3)
                end
            end
        end
    end
    start
end
function sort_by_rem!(descript::Vector{LengthParamDescription}, R = VectorizationBase.REGISTER_SIZE >>> 3)
    i = 1
    N = length(descript)
    # Original version looped while reducing R
    # We only do this once, because that risks misaligning larger objects.
    # E.g., align to 8 doubles, if we have [(l = 10007, id=1, r = 8), (l = 4, id=2, r = 4)]
    # The algorithm would now choose (id=1,id=2) order, but with the loop it would favor (id=2,id=1)
    # 
    # descriptv = @view(descript[1:end])    
    # while R > 1
    # descriptv = @view(descriptv[i:end])
    descriptv = descript
    # update_rem!(descriptv, R)
    sort!(descriptv)
    i = findfirst(d -> !iszero(d.r), descriptv)
    i === nothing && return
    i = combine_param_pairs!(descriptv, R, i)
    i = combine_param_tripples!(descriptv, R, i)
    nothing
        # N < i || return
        # R >>>= 1
    # end
end

function parameter_offsets(descript::Vector{LengthParamDescription})
    sort_by_rem!(descript) # should hopefully align loads and stores reasonably well
    offsets = similar(descript, Int)
    offset = 0
    for d ∈ descript
        # @show d.d.ind, d.l, offset
        offsets[d.d.ind] = offset
        offset += d.l
    end
    offsets
end
@generated function parameter_offsets_noinline(::Val{descript}) where {descript}
    lpd = LengthParamDescription[LengthParamDescription(d)::LengthParamDescription for d ∈ descript]
    offsetvec = parameter_offsets(lpd)
    tup = Expr(:tuple)
    append!(tup.args, offsetvec)
    Expr(:block, Expr(:meta, :noinline), tup)
end
@generated parameter_offsets(::Val{descript}) where {descript} = parameter_offsets_noinline(Val{descript}())
@generated parameter_offset(::Val{descript}, ::Val{I}) where {descript,I} = parameter_offsets_noinline(Val{descript}())[I]

