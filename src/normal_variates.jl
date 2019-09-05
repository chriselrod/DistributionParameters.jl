using PaddedMatrices: DynamicPaddedVector, DynamicPaddedMatrix, AbstractPaddedMatrix, AbstractDynamicPaddedMatrix
using PaddedMatrices: StackPointer, AbstractMutableFixedSizePaddedArray

abstract type AbstractFixedSizeCovarianceMatrix{M,T,R,L} <: PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{M,M,T,R,L} end
mutable struct MutableFixedSizeCovarianceMatrix{M,T,R,L} <: AbstractFixedSizeCovarianceMatrix{M,T,R,L}
    data::NTuple{L,T}
    MutableFixedSizeCovarianceMatrix{M,T,R,L}(::UndefInitializer) where {M,T,R,L} = new{M,T,R,L}()
end
struct PtrFixedSizeCovarianceMatrix{M,T,R,L} <: AbstractFixedSizeCovarianceMatrix{M,T,R,L}
    ptr::Ptr{T}
end
function padl(M, T = Float64)
    Wm1 = VectorizationBase.pick_vector_width(M,T) - 1
    R = (M + Wm1) & ~Wm1
    R, R*M
end
@generated function PtrFixedSizeCovarianceMatrix{M,T}(sp::StackPointer, ::UndefInitializer = undef) where {M,T}
    #    R, L = padl(M,T)
    R = M
    L = R * M
    :(sp + $(sizeof(T)*L), PtrFixedSizeCovarianceMatrix{$M,$T,$R,$L}(pointer(sp, $T)))
end
@generated function MutableFixedSizeCovarianceMatrix{M,T}(::UndefInitializer) where {M,T}
    #    R, L = padl(M,T)
    R = M
    L = R * M
    :(MutableFixedSizeCovarianceMatrix{$M,$T,$R,$L}(undef))
end
@inline FixedSizeCovarianceMatrix(::Val{M}, ::Type{T}) where {M,T} = MutableFixedSizeCovarianceMatrix{M,T}(undef)
@inline FixedSizeCovarianceMatrix(sp::StackPointer, ::Val{M}, ::Type{T}) where {M,T} = PtrFixedSizeCovarianceMatrix{M,T}(sp)


#DynamicCovarianceMatrix{T}(sp::StackPointer, ::UndefInitializer, N) where {T} = DynamicCovarianceMatrix{T}(sp, N)

Base.pointer(A::PtrFixedSizeCovarianceMatrix) = A.ptr
Base.unsafe_convert(::Type{Ptr{T}}, A::PtrFixedSizeCovarianceMatrix{M,T}) where {M,T} = A.ptr
Base.pointer(A::MutableFixedSizeCovarianceMatrix{M,T}) where {M,T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
Base.unsafe_convert(::Type{Ptr{T}}, A::MutableFixedSizeCovarianceMatrix{M,T}) where {M,T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
LinearAlgebra.checksquare(A::AbstractFixedSizeCovarianceMatrix{M}) where {M} = M
@inbounds Base.stride(A::AbstractFixedSizeCovarianceMatrix{M,T,R}, i::Integer) where {M,T,R} = (1, R)[i]
function Base.zeros(::Type{<:MutableFixedSizeCovarianceMatrix{M,T}}) where {M,T}
    A = MutableFixedSizeCovarianceMatrix{M,T}(undef)
    @inbounds for i ∈ 1:PaddedMatrices.full_length(A)
        A[i] = zero(T)
    end
    A
end


struct DynamicCovarianceMatrix{T,ADT <: AbstractDynamicPaddedMatrix{T}} <: AbstractDynamicPaddedMatrix{T} #L
    data::ADT
#    Σ::Symmetric{T,ADT}#Symmetric{T,DynamicPaddedMatrix{T}}
#    ∂Σ::Symmetric{T,ADT}#PaddedMatrix{T}
end
function DynamicCovarianceMatrix{T}(::UndefInitializer, N) where {T}
    DynamicCovarianceMatrix(
        DynamicPaddedMatrix{T}(undef, (N,N))
    )
end
function DynamicCovarianceMatrix{T}(sp::StackPointer, N) where {T}
    sp, data = DynamicPaddedMatrix{T}(sp, (N,N))
    sp, DynamicCovarianceMatrix( data )
end
DynamicCovarianceMatrix{T}(sp::StackPointer, ::UndefInitializer, N) where {T} = DynamicCovarianceMatrix{T}(sp, N)
Base.pointer(A::DynamicCovarianceMatrix) = pointer(A.data)
Base.unsafe_convert(::Type{Ptr{T}}, A::DynamicCovarianceMatrix{T}) where {T} = pointer(A.data)
LinearAlgebra.checksquare(A::DynamicCovarianceMatrix) = size(A.data,2)
Base.stride(A::DynamicCovarianceMatrix, i) = stide(A.data, i)
Base.size(A::DynamicCovarianceMatrix) = (m = size(A.data,2); (m,m))
Base.getindex(A::DynamicCovarianceMatrix, I...) = A.data[I...]
const AbstractCovarianceMatrix{T} = Union{AbstractFixedSizeCovarianceMatrix{M,T} where M,DynamicCovarianceMatrix{T}}


struct MissingDataVector{T,S,M,R,L}#,VT<AbstractVector{T}}
#    bitmask::BitVector
    indices::MutableFixedSizePaddedVector{S,Int,S,S} # note, indices start from 0
#    data::VT
    ∂Σ::MutableFixedSizeCovarianceMatrix{M,T,R,L} # ∂Σ, rather than stack pointer, so that we can guarantee elements are 0
end
struct MissingDataVectorAdjoint{T,S,M,R,L}#,VT}
    mdv::MissingDataVector{T,S,M,R,L}#,VT}
end
function MissingDataVector{T}(bitmask) where {T}
    MissingDataVector{T}(PaddedMatrices.MutableFixedSizePaddedVector, bitmask)
    #=
    N = length(bitmask)
    #    data = zeros(DynamicPaddedMatrix{T}, (N,))
    A = zeros(DynamicPaddedMatrix{T}, (N,N))
    Σ = DynamicCovarianceMatrix{T,typeof(A)}(
        A
    )
    MissingDataVector(
        findall(i -> i == 1, bitmask), Σ#data, Σ
    )
    =#
end
## Not Type Stable
function MissingDataVector{T}(::Type{<:PaddedMatrices.AbstractFixedSizePaddedVector}, bitmask) where {T}
    Σ = zeros(MutableFixedSizeCovarianceMatrix{length(bitmask),T})
    inds = findall(i -> i == 1, bitmask)
    indices = PaddedMatrices.MutableFixedSizePaddedVector{length(inds),Int}(inds)#::PaddedMatrices.MutableFixedSizePaddedVector
    MissingDataVector(
        indices, Σ#data, Σ
    )#::MissingDataVector{T,typeof(Σ),length(inds)}
end
Base.size(::MissingDataVectorAdjoint{T,S,M}) where {T,S,M} = (S,M)


#=
# These versions seem useless, because A and B aren't supposed to be the same size!?!
function mask!(
    B::AbstractPaddedMatrix{T},
    A::AbstractPaddedMatrix{T},
    mrow::BitVector,
    mcol::Vector{<:Integer}
) where {T}
    Nl = LoopVectorization.stride_row(A)
    N = size(A,2)
#    Nl, N = size(d)
    pm = Base.unsafe_convert(Ptr{UInt8}, pointer(mrow.chunks))
    #T_W, T_shift = VectorizationBase.pick_vector_width_shift(T)
    #W = max(8, T_W)
    #shift = max(3, T_shift)
    
    Nr = Nl >> 3
    px = pointer(A)
    pd = pointer(B)
    T_size = sizeof(T)
    @inbounds for nc ∈ mcol
        for n ∈ 0:Nr-1
            vx = SIMDPirates.vload(Vec{8,T}, px + T_size*(nc*Nl+n*8), VectorizationBase.load(pm + n))
            SIMDPirates.vstore!(pd + T_size*(nc*Nl+n*8), vx)
        end
    end
end
function mask(
    A::Union{DynamicCovarianceMatrix{T,ADT2},Symmetric{T,ADT2}},
    mdv::MissingDataVector{T,ADT1}
) where {T, ADT1 <: AbstractDynamicPaddedMatrix{T}, ADT2 <: AbstractDynamicPaddedMatrix{T}}
    ∂Σ = mdv.∂Σ
    mask!(∂Σ, A, mdv.indices)
    ∂Σ
end

function mask!(
    B::Union{DynamicCovarianceMatrix{T,ADT1},Symmetric{T,ADT1}},
    A::Union{DynamicCovarianceMatrix{T,ADT2},Symmetric{T,ADT2}},
    mrow::BitVector, mcol::AbstractVector{<:Integer}
) where {T, ADT1 <: AbstractDynamicPaddedMatrix{T}, ADT2 <: AbstractDynamicPaddedMatrix{T}}
    Nl = LoopVectorization.stride_row(A)
    N = size(A,2)
#    Nl, N = size(d)
    pm = Base.unsafe_convert(Ptr{UInt8}, pointer(mrow.chunks))
    #T_W, T_shift = VectorizationBase.pick_vector_width_shift(T)
    #W = max(8, T_W)
    #shift = max(3, T_shift)
    
    Nr = Nl >> 3
    px = pointer(A.data)
    pd = pointer(B.data)
    T_size = sizeof(T)
    @inbounds for nc ∈ mcol
        for n ∈ (nc >> 3):Nr-1
            vx = SIMDPirates.vload(Vec{8,T}, px + T_size*(nc*Nl+n*8), VectorizationBase.load(pm + n))
            SIMDPirates.vstore!(pd + T_size*(nc*Nl+n*8), vx)
        end
    end
end
=#
#They assume lower triangular symetric
function mask!(
    B::Union{DynamicCovarianceMatrix{T,ADT1},Symmetric{T,ADT1}},
    A::Union{DynamicCovarianceMatrix{T,ADT2},Symmetric{T,ADT2}},
    inds::AbstractVector{<:Integer}
) where {T, ADT1 <: AbstractDynamicPaddedMatrix{T}, ADT2 <: AbstractDynamicPaddedMatrix{T}}
    N = length(inds)
    @inbounds for (ca,cb) ∈ enumerate(inds), ra ∈ ca:N
        rb = inds[ra]
        B[rb,cb] = A[ra,ca]
    end
end
function mask!(
    B::AbstractFixedSizeCovarianceMatrix,
    A::AbstractFixedSizeCovarianceMatrix,
    inds::AbstractVector{<:Integer}
)
    N = length(inds)
    @inbounds for (ca,cb) ∈ enumerate(inds), ra ∈ ca:N
        rb = inds[ra]
        B[rb,cb] = A[ra,ca]
    end
end


function subset!(B::AbstractPaddedMatrix, A::AbstractPaddedMatrix, mdr::MissingDataVector, mdc::MissingDataVector)
    indsr = mdr.indices
    indsc = mdc.indices
    @boundscheck begin
        size(B) == (length(indsr), length(indsc)) || PaddedMatrices.ThrowBoundsError()
#        size(A) == (length(mdr.bitmask), length(mdc.bitmask)) || PaddedMatrices.ThrowBoundsError()
    end

    @inbounds for ics ∈ eachindex(indsc), irs ∈ eachindex(indsr)
        B[irs,ics] = A[indsr[irs],indsc[ics]]
    end

end
function subset!(
    B::Union{DynamicCovarianceMatrix{T,ADT1},Symmetric{T,ADT1}},
    A::Union{DynamicCovarianceMatrix{T,ADT2},Symmetric{T,ADT2}},
    mdv::MissingDataVector
    #mrow::BitVector, mcol::Vector{<:Integer}
) where {T, ADT1 <: AbstractDynamicPaddedMatrix{T}, ADT2 <: AbstractDynamicPaddedMatrix{T}}
#B::AbstractPaddedMatrix, A::AbstractPaddedMatrix, mdr::MissingDataVector, mdc::MissingDataVector)
    inds = mdv.indices
#    indsc = mdc.indices
    N = length(inds)
    @boundscheck begin
        size(B) == (N, N) || PaddedMatrices.ThrowBoundsError()
#        size(A) == (length(mdv.bitmask), length(mdv.bitmask)) || PaddedMatrices.ThrowBoundsError()
    end

    @inbounds for ics ∈ eachindex(inds), irs ∈ ics:N #eachindex(indsr)
        B[irs,ics] = A[inds[irs],inds[ics]]
    end

end
function subset!(
    B::AbstractFixedSizeCovarianceMatrix{S,T},
    A::AbstractFixedSizeCovarianceMatrix{L,T},
    mdv::MissingDataVector{T,S,L}
    #mrow::BitVector, mcol::Vector{<:Integer}
) where {T, S, L}
    inds = mdv.indices
    @inbounds for ics ∈ 1:S, irs ∈ ics:S
        B[irs,ics] = A[inds[irs],inds[ics]]
    end
end

function Base.getindex( A::DynamicCovarianceMatrix{T}, mdv::MissingDataVector ) where {T}
    B = DynamicCovarianceMatrix{T}( undef, length(mdv.indices) )
    @inbounds subset!( B, A, mdv )
    B
end
function Base.getindex( sp::StackPointer, A::DynamicCovarianceMatrix{T}, mdv::MissingDataVector ) where {T}
    sp, B = DynamicCovarianceMatrix{T}( sp, length(mdv.indices) )
    @inbounds subset!( B, A, mdv )
    sp, B
end
function PaddedMatrices.∂getindex( A::DynamicCovarianceMatrix{T}, mdv::MissingDataVector ) where {T}
    B = DynamicCovarianceMatrix{T}( undef, length(mdv.indices) )
    @inbounds subset!( B, A, mdv )
    B, MissingDataVectorAdjoint(mdv)
end
function PaddedMatrices.∂getindex( sp::StackPointer, A::DynamicCovarianceMatrix{T}, mdv::MissingDataVector ) where {T}
    sp, B = DynamicCovarianceMatrix{T}( sp, length(mdv.indices) )
    @inbounds subset!( B, A, mdv )
    sp, (B, MissingDataVectorAdjoint(mdv))
end

function Base.getindex( A::AbstractFixedSizeCovarianceMatrix{L,T}, mdv::MissingDataVector{T,S,L} ) where {T,S,L}
    B = MutableFixedSizeCovarianceMatrix{S,T}(undef)
    @inbounds subset!( B, A, mdv )
    B
end
function Base.getindex( sp::StackPointer, A::AbstractFixedSizeCovarianceMatrix{L,T}, mdv::MissingDataVector{T,S,L} ) where {T,S,L}
    sp, B = PtrFixedSizeCovarianceMatrix{S,T}( sp )
    @inbounds subset!( B, A, mdv )
    sp, B
end
function PaddedMatrices.∂getindex( A::AbstractFixedSizeCovarianceMatrix{L,T}, mdv::MissingDataVector{T,S,L} ) where {T,S,L}
    B = MutableFixedSizeCovarianceMatrix{S,T}(undef)
    @inbounds subset!( B, A, mdv )
    B, MissingDataVectorAdjoint(mdv)
end
function PaddedMatrices.∂getindex( sp::StackPointer, A::AbstractFixedSizeCovarianceMatrix{L,T}, mdv::MissingDataVector{T,S,L} ) where {T,S,L}
    sp, B = PtrFixedSizeCovarianceMatrix{S,T}( sp )
    @inbounds subset!( B, A, mdv )
    sp, (B, MissingDataVectorAdjoint(mdv))
end

function Base.getindex(
    a::AbstractMutableFixedSizePaddedVector{M,T},
    mdv::MissingDataVector{T,S,M}
) where {T,S,M}
    inds = mdv.indices
    b = MutableFixedSizePaddedVector{S,T}(undef)
    @inbounds for s ∈ 1:S
        b[s] = a[inds[s]]
    end
    b
end
function PaddedMatrices.∂getindex(
    a::AbstractMutableFixedSizePaddedVector{M,T},
    mdv::MissingDataVector{T,S,M}
) where {T,S,M}
    inds = mdv.indices
    b = MutableFixedSizePaddedVector{S,T}(undef)
    @inbounds for s ∈ 1:S
        b[s] = a[inds[s]]
    end
    b, MissingDataVectorAdjoint(mdv)
end
function Base.getindex(
    sp::StackPointer,
    a::AbstractMutableFixedSizePaddedVector{M,T},
    mdv::MissingDataVector{T,S,M}
) where {T,S,M}
    inds = mdv.indices
    sp, b = PaddedMatrices.PtrVector{S,T}(sp)
    @inbounds for s ∈ 1:S
        b[s] = a[inds[s]]
    end
    sp, b
end
function PaddedMatrices.∂getindex(
    sp::StackPointer,
    a::AbstractMutableFixedSizePaddedVector{M,T},
    mdv::MissingDataVector{T,S,M}
) where {T,S,M}
    inds = mdv.indices
    sp, b = PaddedMatrices.PtrVector{S,T}(sp)
    @inbounds for s ∈ 1:S
        b[s] = a[inds[s]]
    end
    sp, (b, MissingDataVectorAdjoint(mdv))
end
@generated function Base.getindex(
    sp::StackPointer,
    a::NTuple{K,V},
    mdv::MissingDataVector{T,S,M}
) where {T,K,S,M,V<:PaddedMatrices.AbstractMutableFixedSizePaddedVector{M,T}}
    quote
        inds = mdv.indices
        Base.Cartesian.@nexprs $K k -> (a_k = @inbounds a[k]; (sp, b_k) = PaddedMatrices.PtrVector{$S,$T}(sp))
#        @show typeof(a_1)
#        @show typeof(b_1)
#        println("P2: $P2\n")
        for s ∈ 1:$S
            #        @inbounds for p ∈ 1:$P2
            i = inds[s]
            Base.Cartesian.@nexprs $K k-> (b_k[s] = a_k[i])
        end
        sp, (Base.Cartesian.@ntuple $K b)
    end
end
@generated function PaddedMatrices.∂getindex(
    sp::StackPointer,
    a::NTuple{K,V},
    mdv::MissingDataVector{T,S,M}
) where {T,K,S,M,V<:PaddedMatrices.AbstractMutableFixedSizePaddedVector{M,T}}
    quote
        inds = mdv.indices
        Base.Cartesian.@nexprs $K k -> (a_k = @inbounds a[k]; (sp, b_k) = PaddedMatrices.PtrVector{$S,$T}(sp))
        @inbounds for s ∈ 1:$S
            i = inds[s]
            Base.Cartesian.@nexprs $K k-> (b_k[s] = a_k[i])
        end
        sp, ((Base.Cartesian.@ntuple $K b), MissingDataVectorAdjoint(mdv))
    end
end


function Base.:*(A::AbstractCovarianceMatrix, mdv::MissingDataVectorAdjoint)
    ∂Σ = mdv.mdv.∂Σ
    mask!(∂Σ, A, mdv.mdv.indices)
    ∂Σ
end
@generated function Base.:*(
    a::NTuple{K,V},
    mdv::MissingDataVectorAdjoint{T,S,M}
) where {K,T,S,M,V<:PaddedMatrices.AbstractMutableFixedSizePaddedVector{S,T}}
    quote
        Base.Cartesian.@nexprs $K k -> (b_k = zeros(MutableFixedSizePaddedVector{$M,$T}); a_k = a[k])
        inds = mdv.mdv.indices
        @inbounds for s ∈ 1:$S
            i = inds[s]
            Base.Cartesian.@nexprs $K k -> b_k[i] = a_k[s]
        end
        Base.Cartesian.@ntuple $K b
    end
end
@generated function Base.:*(
    sp::StackPointer,
    a::NTuple{K,V},
    mdv::MissingDataVectorAdjoint{T,S,M}
) where {K,T,S,M,V<:PaddedMatrices.AbstractMutableFixedSizePaddedVector{S,T}}
    Wm1 = VectorizationBase.pick_vector_width(M,T) - 1
#    MR = (M + Wm1) & ~Wm1
    total_length = (K*M + Wm1) & + ~Wm1
    quote
        # we zero the used data in a single loop.
        zero_out = PtrVector{$total_length,$T,$total_length,$total_length}(pointer(sp, $T))
        @inbounds @simd ivdep for i ∈ 1:$total_length
            zero_out[i] = zero($T)
        end
        
        Base.Cartesian.@nexprs $K k -> begin
            b_k = PtrVector{$M,$T,$M,$M}(pointer(sp,$T) + $(sizeof(T)*M) * (k-1))
            a_k = a[k]
        end
        inds = mdv.mdv.indices
        @inbounds for s ∈ 1:$S
            i = inds[s]
            Base.Cartesian.@nexprs $K k -> b_k[i] = a_k[s]
        end
        sp + $(VectorizationBase.align(sizeof(T)*M*K)), (Base.Cartesian.@ntuple $K b)
    end
end

#=

"""
Variadic function. Given K groups (as tuples) of present indices...
"""
MissingDataMatrix(missingness...) = MissingDataMatrix(missingness)
@generated function MissingDataMatrix(
    missingness::KT
) where {KT}
    K = length(KT.parameters)

    quote
        patterns = Vector{Int}
        for k ∈ 1:K
            
        end
    end
end
# Type unstable, but is only run for data organization.
function MissingDataMatrix(missingness::Tuple)
    K = length(missingness)
    patterns = Union{UnitRange{Int},Vector{Int}}[]
    pattern_inds = MutableFixedSizePaddedVector{K,Int,K,K}(undef)
    for k ∈ 1:K
        m = missingness[k]
        ind = findfirst(x -> x == m, patterns)
        if ind == nothing
            push!(patterns, m)
            pattern_inds[k] = length(patterns)
        else
            pattern_inds[k] = ind
        end
    end
    quit_vector = Vector{Int}(undef, length(patterns))
    quit_inds = Vector{Int}(undef, length(patterns) - 1)
    quit_vector[1] = last(pattern_inds)
    i = K-1
    for k ∈ K-1:-1:1
        pi = pattern_inds[k]
        if pi ∉ quit_vector
            
        end
    end
    
    # Now, which are downstream?
    downstream = [ last(pattern_inds) ]
    down_stream_groups = Vector{Vector{Int}}(undef, K-1)
    for k ∈ K-1:-1:1
        
    end
end
=#

#=
struct ∂MultivariateNormalVariate{T,ADPM<:AbstractDynamicPaddedMatrix{T}} <: AbstractMatrix{T}
    data::ADPM
end
Base.size(A::∂MultivariateNormalVariate) = size(A.data)
Base.getindex(A::∂MultivariateNormalVariate, I...) = Base.getindex(A.data, I...)

struct MultivariateNormalVariate{T,ADPM<:AbstractDynamicPaddedMatrix{T}} <: AbstractMatrix{T}
    data::ADPM
    δ::ADPM
    Σ⁻¹δ::∂MultivariateNormalVariate{T,ADPM}
end
Base.size(A::MultivariateNormalVariate) = size(A.data)
Base.getindex(A::MultivariateNormalVariate, I...) = Base.getindex(A.data, I...)

# Data stored in lower triangle.
Base.size(A::DynamicCovarianceMatrix) = size(A.data)
Base.getindex(A::DynamicCovarianceMatrix, I...) = Base.getindex(A.data, I...)

LinearAlgebra.cholesky!(Σ::DynamicCovarianceMatrix) = LinearAlgebra.LAPACK.potrf!('L', Σ.Σ.data)


function Base.:\(U::UpperTriangular{T,Matrix{T}}, Y::MultivariateNormalVariate{T}) where {T}
    δ = Y.δ
    # Σ⁻¹δ = Y.Σ⁻¹δ
    # copyto!(Σ⁻¹δ, δ)
    LinearAlgebra.LAPACK.trtrs!('L', 'N', 'N', U.data, δ)
end

function Base.:\(U::Cholesky{T,Matrix{T}}, Y::MultivariateNormalVariate{T}) where {T}
    δ = Y.δ
    Σ⁻¹δ = Y.Σ⁻¹δ.data
    copyto!(Σ⁻¹δ, δ)
    LinearAlgebra.LAPACK.potrs!('L', U.factors, Σ⁻¹δ)
end
function Base.:\(Σ::DynamicCovarianceMatrix{T}, Y::MultivariateNormalVariate{T}) where {T <: BLAS.BlasFloat}
    LinearAlgebra.LAPACK.potrf!('L', Σ.Σ.data)
    δ = Y.δ
    Σ⁻¹δ = Y.Σ⁻¹δ.data
    copyto!(Σ⁻¹δ, δ)
    LinearAlgebra.LAPACK.potrs!('L', Σ.Σ.data, Σ⁻¹δ)
end
=#

@generated function DynamicCovarianceMatrix(
                rhos::PaddedMatrices.AbstractFixedSizePaddedVector{K,T}, L::StructuredMatrices.AbstractLowerTriangularMatrix{K},
                times::ConstantFixedSizePaddedVector{nT}, workspace
            ) where {K,T,nT}
    Wm1 = VectorizationBase.pick_vector_width(nT, T) - 1
    nTl = (nT + Wm1) & ~Wm1
    quote
        # nT = length(times) + 1
        # K = size(L,1)
        ARs = workspace.ARs
#        @show eltype(ARs), eltype(rhos)
        @fastmath @inbounds for k ∈ 1:$K
            ρ = rhos[k]
            logρ = $(T == Float64 ? :(ccall(:log,Float64,(Float64,),ρ))  :  :(Base.log(ρ)))
            for tc ∈ 1:nT
                for tr ∈ 1:tc-1
                    ARs[tr,tc,k] = ARs[tc,tr,k]
                end
                ARs[tc,tc,k] = one(T)
                for tr ∈ tc+1:nT
                    deltatimes = times[tr] - times[tc]
                    logrhot = deltatimes*logρ
                    ARs[tr,tc,k] = $(T == Float64 ? :(ccall(:exp,Float64,(Float64,),logrhot)) : :(Base.exp(logrhot)))
                end
            end
            # ARs[:,:,k] .= AutoregressiveMatrix(rhos[k], δₜ)
        end
        # Base.Cartesian.@nexprs $K k -> AR_k = ConstantFixedSizePaddedMatrix(AutoregressiveMatrix(rhos[k], δₜ))
        # ARs = [ConstantFixedSizePaddedMatrix(AutoregressiveMatrix(rho, δₜ)) for rho ∈ rhos]
        Sigfull = workspace.Sigfull
        # ∂Sig∂L
        @inbounds begin
        Base.Cartesian.@nexprs $K kc -> begin
            for kr ∈ kc-1:K-1
                Base.Cartesian.@nexprs kc j -> l_j = L[kr+1,j] * L[kc,j]
                # Sigview = @view Sigfull[(1:$nTl) .+ $nT*kr, (1:$nT) .+ $nT*(kc-1)]
                sigrow, sigcol = $nT*kr, $nT*(kc-1)
#                sigrow, sigcol = $nT*(kc-1), $nT*kr # transpose it
                for tc ∈ 1:$nT
                    @simd ivdep for tr ∈ 1:$nTl
                        ari = l_1 * ARs[tr, tc, 1]
                        Base.Cartesian.@nexprs kc-1 j -> ari = muladd(l_{j+1}, ARs[tr, tc, j+1], ari)
                        Sigfull[ tr + sigrow, tc + sigcol] = ari
                    end
                end
            end
        end
        end
        DynamicCovarianceMatrix(Sigfull)
    end
end


struct Covariance_LAR_AR_Adjoint{K, nT, T, nTP, L, MFPD<:AbstractMutableFixedSizePaddedArray{Tuple{nT,nT,K},T,3,nTP,L}, LKJ_T <: StructuredMatrices.AbstractLowerTriangularMatrix}
    ∂ARs::MFPD#MutableFixedSizePaddedArray{Tuple{nT,nT,K},T,3,nTP,L}
    LKJ::LKJ_T
end
struct Covariance_LAR_L_Adjoint{K, nT, T, nTP, L, MFPD<:AbstractMutableFixedSizePaddedArray{Tuple{nT,nT,K},T,3,nTP,L}, LKJ_T <: StructuredMatrices.AbstractLowerTriangularMatrix}
    ARs::MFPD#MutableFixedSizePaddedArray{Tuple{nT,nT,K},T,3,nTP,L}
    LKJ::LKJ_T
end

@generated function ∂DynamicCovarianceMatrix(
    rhos::PaddedMatrices.AbstractFixedSizePaddedVector{K,T},
    L::StructuredMatrices.AbstractLowerTriangularMatrix{K,T},
    times::Union{<:AbstractFixedSizePaddedVector{nT},PaddedMatrices.StaticUnitRange{nT}},
    workspace, ::Val{(true,true)}
) where {K,T,nT}
    # We will assume rho > 0
    Wm1 = VectorizationBase.pick_vector_width(nT, T) - 1
    nTl = (nT + Wm1) & ~Wm1
    quote
        # nT = length(times) + 1
        # K = size(L,1)
        ARs = workspace.ARs
        ∂ARs = workspace.∂ARs
        @inbounds @fastmath for k ∈ 1:$K
            ρ = rhos[k]
            # We want to use Base.log, and not fastmath log.
            
            logρ = $(T == Float64 ? :(ccall(:log,Float64,(Float64,),ρ))  :  :(Base.log(ρ)))

            for tc ∈ 1:nT
                for tr ∈ 1:tc-1
                    ARs[tr,tc,k] = ARs[tc,tr,k]
                    ∂ARs[tr,tc,k] = ∂ARs[tc,tr,k]
                end
                ARs[tc,tc,k] = one(T)
                ∂ARs[tc,tc,k] = zero(T)
                for tr ∈ tc+1:nT
                    deltatimes = times[tr] - times[tc]
                    #                    rhot = rho^(deltatimes - one(T))
                    # We want to specifically use Base.exp, and not fastmath exp
                    logrhot = (deltatimes - one($T))*logρ
                    rhot = $(T == Float64 ? :(ccall(:exp,Float64,(Float64,),logrhot)) : :(Base.exp(logrhot)))
                    ARs[tr,tc,k] = ρ*rhot
                    ∂ARs[tr,tc,k] = deltatimes*rhot
                end
            end
            # ARs[:,:,k] .= AutoregressiveMatrix(rhos[k], δₜ)
        end
        Sigfull = workspace.Sigfull
        # ∂Sig∂L
        @inbounds begin
        Base.Cartesian.@nexprs $K kc -> begin
            for kr ∈ kc-1:K-1
                Base.Cartesian.@nexprs kc j -> l_j = L[kr+1,j] * L[kc,j]
                # Sigview = @view Sigfull[(1:$nTl) .+ $nT*kr, (1:$nT) .+ $nT*(kc-1)]
                sigrow, sigcol = $nT*kr, $nT*(kc-1)
#                sigrow, sigcol = $nT*(kc-1), $nT*kr # transpose it
                for tc ∈ 1:$nT
                    @simd ivdep for tr ∈ 1:$nTl
                        ari = l_1 * ARs[tr, tc, 1]
                        Base.Cartesian.@nexprs kc-1 j -> ari = muladd(l_{j+1}, ARs[tr, tc, j+1], ari)
                        Sigfull[ tr + sigrow, tc + sigcol] = ari
                    end
                end
            end
        end
        end
        DynamicCovarianceMatrix(Sigfull), Covariance_LAR_AR_Adjoint(∂ARs,L), Covariance_LAR_L_Adjoint(ARs,L)
    end
end

@generated function CovarianceMatrix(
    sp::StackPointer,
    rhos::PaddedMatrices.AbstractFixedSizePaddedVector{K,T},
    L::StructuredMatrices.AbstractLowerTriangularMatrix{K,T},
    times::Union{<:AbstractFixedSizePaddedVector{nT},PaddedMatrices.StaticUnitRange{nT}},
#    missing::MissingDataMatrix,
    #    ::Val{(true,true)},
    ::Val{SIMD} = Val{true}()
) where {K,T,nT,SIMD}
    # We will assume rho > 0
    W, Wshift = VectorizationBase.pick_vector_width_shift(nT, T)
    Wm1 = W - 1
    V = Vec{W,T}
    nTl = (nT + Wm1) & ~Wm1
    KT = K*nT
    # We want it to be a multiple of 8, and we don't want nTl bleeding over the edge
    KTR = ( KT + nTl - nT + Wm1 ) & ~Wm1
    T_size = sizeof(T)
    quote
        # only Sigfull escapes, so we allocate it first
        # and return the stack pointer pointing to its end.
        sp, Sigfull = PtrFixedSizeCovarianceMatrix{$KT,$T}(sp)
        ARs = PaddedMatrices.PtrArray{Tuple{$nT,$nT,$K},$T,3,$nTl}(pointer(sp, $T))

        ptr_time = pointer(times)
        ptr_ARs = pointer(ARs)
        @inbounds for k ∈ 1:$K
            ρ = rhos[k]
            # We want to use Base.log, and not fastmath log.
            logρ = $(T == Float64 ? :(ccall(:log,Float64,(Float64,),ρ))  :  :(Base.log(ρ)))
            $(SIMD ? quote
            vρ = SIMDPirates.vbroadcast($V, ρ)
            vlogρ = SIMDPirates.vbroadcast($V, logρ)
            for tcouter ∈ 0:$((nTl>>Wshift)-1)
                for tcinner ∈ 1:$W
                    Wtcouter = $W*tcouter
                    tc = tcinner + Wtcouter
                    tc > $nT && break
                    for tr ∈ 1:Wtcouter
                        ARs[tr,tc,k] = ARs[tc,tr,k]
                    end
                    vtime_tc = SIMDPirates.vbroadcast($V, times[tc])
                    for i ∈ tcouter:$((nTl>>Wshift)-1)
                        vtimes_tr = SIMDPirates.vload($V, ptr_time + i * $(T_size*W))
                        vδt = SIMDPirates.vabs(SIMDPirates.vsub(vtimes_tr, vtime_tc))
                        vδtm1 = SIMDPirates.vsub(vδt, SIMDPirates.vbroadcast($V,one($T)))
                        vρt = SLEEFPirates.exp(SIMDPirates.vmul(vδtm1, vlogρ))
                        AR_offset = (k-1)*$(T_size*nTl*nT) + (tc-1)*$(T_size*nTl) + i*$(W*T_size)

                        SIMDPirates.vstore!(ptr_ARs + AR_offset, SIMDPirates.vmul(vρ,vρt))
                    end
                end
              end
              end : quote
            for tc ∈ 1:$nT
                for tr ∈ 1:tc-1
                    ARs[tr,tc,k] = ARs[tc,tr,k]
                end
                ARs[tc,tc,k] = one(T)
                for tr ∈ tc+1:$nT
                    deltatimes = times[tr] - times[tc]
                    #                    rhot = rho^(deltatimes - one(T))
                    # We want to specifically use Base.exp, and not fastmath exp
                    logrhot = (deltatimes - one($T))*logρ
                    rhot = $(T == Float64 ? :(ccall(:exp,Float64,(Float64,),logrhot)) : :(Base.exp(logrhot)))
                    ARs[tr,tc,k] = ρ*rhot
                end
              end
              end )
        end
        # ∂Sig∂L
        @inbounds begin
        Base.Cartesian.@nexprs $K kc -> begin
            for kr ∈ kc-1:K-1
                Base.Cartesian.@nexprs kc j -> l_j = L[kr+1,j] * L[kc,j]
                # Sigview = @view Sigfull[(1:$nTl) .+ $nT*kr, (1:$nT) .+ $nT*(kc-1)]
                sigrow, sigcol = $nT*kr, $nT*(kc-1)
#                sigrow, sigcol = $nT*(kc-1), $nT*kr # transpose it
                for tc ∈ 0:$(nT-1)
                    @vectorize for tr ∈ 1:$nT
                        ari = l_1 * ARs[tr + tc * $nTl]
                        Base.Cartesian.@nexprs kc-1 j -> ari = muladd(l_{j+1}, ARs[tr + tc*$nTl + j*$(nTl*nT)], ari)
                        Sigfull[ tr + sigrow + (tc + sigcol)*$KT] = ari 
                    end
                end
            end
        end
        end
#        sp, (Sigfull,(∂ARs,L), (ARs,L))
        sp, Sigfull
    end
end
@generated function ∂CovarianceMatrix(
    sp::StackPointer,
    rhos::PaddedMatrices.AbstractFixedSizePaddedVector{K,T},
    L::StructuredMatrices.AbstractLowerTriangularMatrix{K,T},
    times::Union{<:AbstractFixedSizePaddedVector{nT},PaddedMatrices.StaticUnitRange{nT}},
#    missing::MissingDataMatrix,
    ::Val{(true,true)},::Val{SIMD} = Val{true}()
) where {K,T,nT,SIMD}
    # We will assume rho > 0
    W, Wshift = VectorizationBase.pick_vector_width_shift(nT, T)
    Wm1 = W - 1
    V = Vec{W,T}
    nTl = (nT + Wm1) & ~Wm1
    KT = K*nT
    # We want it to be a multiple of 8, and we don't want nTl bleeding over the edge
    KTR = ( KT + nTl - nT + Wm1 ) & ~Wm1
    T_size = sizeof(T)
#    sym = false
    sym = true
    ARquote = if SIMD && sym
        quote
            vρ = SIMDPirates.vbroadcast($V, ρ)
            vlogρ = SIMDPirates.vbroadcast($V, logρ)
            for tcouter ∈ 0:$((nTl>>Wshift)-1)
                for tcinner ∈ 1:$W
                    Wtcouter = $W*tcouter
                    tc = tcinner + Wtcouter
                    tc > $nT && break
                    for tr ∈ 1:Wtcouter
                        ARs[tr,tc,k] = ARs[tc,tr,k]
                        ∂ARs[tr,tc,k] = ∂ARs[tc,tr,k]
                    end
                    vtime_tc = SIMDPirates.vbroadcast($V, times[tc])
                    for i ∈ tcouter:$((nTl>>Wshift)-1)
                        vtimes_tr = SIMDPirates.vload($V, ptr_time + i * $(T_size*W))
                        vδt = SIMDPirates.vabs(SIMDPirates.vsub(vtimes_tr, vtime_tc))
                        vδtm1 = SIMDPirates.vsub(vδt, SIMDPirates.vbroadcast($V,one($T)))
                        vρt = SLEEFPirates.exp(SIMDPirates.vmul(vδtm1, vlogρ))
                        AR_offset = (k-1)*$(T_size*nTl*nT) + (tc-1)*$(T_size*nTl) + i*$(W*T_size)

                        SIMDPirates.vstore!(ptr_ARs + AR_offset,
                                             SIMDPirates.vmul(vρ,vρt))
                        SIMDPirates.vstore!(ptr_∂ARs + AR_offset,
                                            SIMDPirates.vmul(vδt,vρt))
                    end
                end
              end
        end
    elseif SIMD
        quote
            vρ = SIMDPirates.vbroadcast($V, ρ)
            vlogρ = SIMDPirates.vbroadcast($V, logρ)
            @inbounds for c ∈ 0:$(nT-1)
                #                time_c = times[c+1]
                #                offset = (k-1)*$(nTl*nT) + c*$nTl
                offset = (k-1)*$(T_size*nTl*nT) + c*$(T_size*nTl)
                vtime_tc = SIMDPirates.vbroadcast($V, times[c+1])
                for r ∈ 0:$((nTl >> Wshift)-1)
                    vtimes_tr = SIMDPirates.vload($V, ptr_time + r * $(T_size*W))
                    vδt = SIMDPirates.vabs(SIMDPirates.vsub(vtimes_tr, vtime_tc))
                    vδtm1 = SIMDPirates.vsub(vδt, SIMDPirates.vbroadcast($V,one($T)))
                    vρt = SLEEFPirates.exp(SIMDPirates.vmul(vδtm1, vlogρ))
                    AR_offset = offset + r*$(W*T_size)

                    SIMDPirates.vstore!(ptr_ARs + AR_offset,
                                        SIMDPirates.vmul(vρ,vρt))
                    SIMDPirates.vstore!(ptr_∂ARs + AR_offset,
                                        SIMDPirates.vmul(vδt,vρt))
                end
#                LoopVectorization.@vvectorize for r ∈ 1:$nTl
#                    times_r = times[r]
#                    δt = SIMDPirates.vabs(times_r - time_c)
#                    δtm1 = δt - 1
#                    ρtm1 = SLEEFPirates.exp(δtm1 * logρ)
#                    ARs[r + offset] = ρtm1 * ρ
#                    ∂ARs[r + offset] = ρtm1 * δt
#                end
            end
        end
    else
        quote
            for tc ∈ 1:$nT
                for tr ∈ 1:tc-1
                    ARs[tr,tc,k] = ARs[tc,tr,k]
                    ∂ARs[tr,tc,k] = ∂ARs[tc,tr,k]
                end
                ARs[tc,tc,k] = one(T)
                ∂ARs[tc,tc,k] = zero(T)
                for tr ∈ tc+1:$nT
                    deltatimes = times[tr] - times[tc]
                    #                    rhot = rho^(deltatimes - one(T))
                    # We want to specifically use Base.exp, and not fastmath exp
                    logrhot = (deltatimes - one($T))*logρ
                    rhot = $(T == Float64 ? :(ccall(:exp,Float64,(Float64,),logrhot)) : :(Base.exp(logrhot)))
                    ARs[tr,tc,k] = ρ*rhot
                    ∂ARs[tr,tc,k] = deltatimes*rhot
                end
            end
        end
    end
        
    quote
        # nT = length(times) + 1
        # K = size(L,1)
#       U = missing.unique_patterns
#        ARs = workspace.ARs
        #        ∂ARs = workspace.∂ARs
        sp,  ARs = PaddedMatrices.PtrArray{Tuple{$nT,$nT,$K},$T,3,$nTl}(sp)
        sp, ∂ARs = PaddedMatrices.PtrArray{Tuple{$nT,$nT,$K},$T,3,$nTl}(sp)
        #        @inbounds @fastmath for k ∈ 1:$K
        ptr_time = pointer(times)
        ptr_ARs = pointer(ARs)
        ptr_∂ARs = pointer(∂ARs)
#        @time for i ∈ 1:100000
        @inbounds for k ∈ 1:$K
            ρ = rhos[k]
            # We want to use Base.log, and not fastmath log.
            logρ = $(T == Float64 ? :(ccall(:log,Float64,(Float64,),ρ))  :  :(Base.log(ρ)))
            $ARquote
        end
#        end
        sp, Sigfull = PtrFixedSizeCovarianceMatrix{$KT,$T}(sp)
        # ∂Sig∂L
        @inbounds begin
        Base.Cartesian.@nexprs $K kc -> begin
            for kr ∈ kc-1:K-1
                Base.Cartesian.@nexprs kc j -> l_j = L[kr+1,j] * L[kc,j]
                # Sigview = @view Sigfull[(1:$nTl) .+ $nT*kr, (1:$nT) .+ $nT*(kc-1)]
                sigrow, sigcol = $nT*kr, $nT*(kc-1)
#                sigrow, sigcol = $nT*(kc-1), $nT*kr # transpose it
                for tc ∈ 0:$(nT-1)
                    @vectorize for tr ∈ 1:$nT
                        ari = l_1 * ARs[tr + tc * $nTl]
                        Base.Cartesian.@nexprs kc-1 j -> ari = muladd(l_{j+1}, ARs[tr + tc*$nTl + j*$(nTl*nT)], ari)
                        Sigfull[ tr + sigrow + (tc + sigcol)*$KT] = ari 
                    end
                end
            end
        end
        end
#        println("Calculating Covariance Matrix, ∂ARs:")
#        display(∂ARs)
#        println("Calculating Covariance Matrix, L:")
#        display(L)
#        @show Array(∂ARs)[1:6,1:6,:]
#=        @inbounds begin
        Base.Cartesian.@nexprs $K kc -> begin
            for kr ∈ kc-1:K-1
                Base.Cartesian.@nexprs kc j -> l_j = L[kr+1,j] * L[kc,j]
                # Sigview = @view Sigfull[(1:$nTl) .+ $nT*kr, (1:$nT) .+ $nT*(kc-1)]
                sigrow, sigcol = $nT*kr, $nT*(kc-1)
#                sigrow, sigcol = $nT*(kc-1), $nT*kr # transpose it
                for tc ∈ 1:$nT
                    @simd ivdep for tr ∈ 1:$nTl
                        ari = l_1 * ARs[tr, tc, 1]
                        Base.Cartesian.@nexprs kc-1 j -> ari = muladd(l_{j+1}, ARs[tr, tc, j+1], ari)
                        Sigfull[ tr + sigrow, tc + sigcol] = ari
                    end
                end
            end
        end
        end=#
#        sp, (Sigfull,(∂ARs,L), (ARs,L))
        sp, (Sigfull, Covariance_LAR_AR_Adjoint(∂ARs,L), Covariance_LAR_L_Adjoint(ARs,L))
#        sp + (252^2*24), (Sigfull, Covariance_LAR_AR_Adjoint(∂ARs,L), Covariance_LAR_L_Adjoint(ARs,L))
    end
end

@generated function Base.:*(C::AbstractMatrix{T},
                adj::Covariance_LAR_AR_Adjoint{K, nT, T, nTP}
                ) where {K, nT, T, nTP}
    # C is a DynamicCovarianceMatrix
    Wm1 = VectorizationBase.pick_vector_width(T)-1 
    KL = (K+Wm1) & ~Wm1
#    outtup = Expr(:tuple, [Expr(:call,:*,2,Symbol(:κ_,k)) for k ∈ 1:K]..., [zero(T) for k ∈ K+1:KL]...)
    outtup = Expr(:tuple, [Symbol(:κ_,k) for k ∈ 1:K]..., [zero(T) for k ∈ K+1:KL]...)
    quote
        $(Expr(:meta,:inline))
        # Calculate all in 1 pass?
        # ∂C∂ρ = MutableFixedSizePaddedVector{K,T}(undef)
        # Cstride = stride(C,2)
        L = adj.LKJ
        ∂ARs = adj.∂ARs
#        println("Reverse pass, partial cov:")
#        display(C)
#        println("Reverse pass, using ∂ARs:")
#        display(∂ARs)
#        println("Reverse pass, L:")
#        display(L)
#        @show Array(∂ARs)[1:6,1:6,:]

        Base.Cartesian.@nexprs $K j -> κ_j = zero($T)
        @inbounds begin
            Base.Cartesian.@nexprs $K kc -> begin
                # Diagonal block
#                kr = kc-1
                Base.Cartesian.@nexprs kc j -> l_j = L[kc,j]^2
                ccol = crow = $nT*(kc-1)
                for tc ∈ 1:$nT
                    tco = tc + ccol
                    for tr ∈ tc+1:$nT
                        cij = C[tr + crow, tco]
                        Base.Cartesian.@nexprs kc j -> begin
                            κ_j = Base.FastMath.add_fast(Base.FastMath.mul_fast(cij,l_j,∂ARs[tr,tc,j]), κ_j)
                        end
                    end
                end

                for kr ∈ kc:$(K-1)
                    Base.Cartesian.@nexprs kc j -> l_j = L[kc,j]*L[kr+1,j]
                    crow = $nT*kr; ccol = $nT*(kc-1)
                    for tc ∈ 1:$nT
                        tco = tc + ccol
                        for tr ∈ 1:$nT
                            cij = C[tr + crow, tco]
                            Base.Cartesian.@nexprs kc j -> begin
                                κ_j = Base.FastMath.add_fast(Base.FastMath.mul_fast(cij,l_j,∂ARs[tr,tc,j]), κ_j)
                            end
                        end
                    end
                end
            end
        end
        ConstantFixedSizePaddedVector{$K,$T,$KL,$KL}( $outtup )'
    end
end
@generated function Base.:*(
    sp::StackPointer,
    C::AbstractMatrix{T},
    adj::Covariance_LAR_AR_Adjoint{K, nT, T, nTP}
) where {K, nT, T, nTP}
    # C is a DynamicCovarianceMatrix
    Wm1 = VectorizationBase.pick_vector_width(T)-1 
    KL = (K+Wm1) & ~Wm1
#    outtup = Expr(:tuple, [Expr(:call,:*,2,Symbol(:κ_,k)) for k ∈ 1:K]..., [zero(T) for k ∈ K+1:KL]...)
#    outtup = Expr(:tuple, [Symbol(:κ_,k) for k ∈ 1:K]..., [zero(T) for k ∈ K+1:KL]...)
    quote
        $(Expr(:meta,:inline))
        # Calculate all in 1 pass?
        # ∂C∂ρ = MutableFixedSizePaddedVector{K,T}(undef)
        # Cstride = stride(C,2)
        L = adj.LKJ
        ∂ARs = adj.∂ARs
#        println("Reverse pass, partial cov:")
#        display(C)
#        println("Reverse pass, using ∂ARs:")
#        display(∂ARs)
#        println("Reverse pass, L:")
#        display(L)
#        @show Array(∂ARs)[1:6,1:6,:]

        Base.Cartesian.@nexprs $K j -> κ_j = zero($T)
        @inbounds begin
            Base.Cartesian.@nexprs $K kc -> begin
                # Diagonal block
#                kr = kc-1
                Base.Cartesian.@nexprs kc j -> l_j = L[kc,j]^2
                ccol = crow = $nT*(kc-1)
                for tc ∈ 1:$nT
                    tco = tc + ccol
                    for tr ∈ tc+1:$nT
                        cij = C[tr + crow, tco]
                        Base.Cartesian.@nexprs kc j -> begin
                            κ_j = Base.FastMath.add_fast(Base.FastMath.mul_fast(cij,l_j,∂ARs[tr,tc,j]), κ_j)
                        end
                    end
                end

                for kr ∈ kc:$(K-1)
                    Base.Cartesian.@nexprs kc j -> l_j = L[kc,j]*L[kr+1,j]
                    crow = $nT*kr; ccol = $nT*(kc-1)
                    for tc ∈ 1:$nT
                        tco = tc + ccol
                        for tr ∈ 1:$nT
                            cij = C[tr + crow, tco]
                            Base.Cartesian.@nexprs kc j -> begin
                                κ_j = Base.FastMath.add_fast(Base.FastMath.mul_fast(cij,l_j,∂ARs[tr,tc,j]), κ_j)
                            end
                        end
                    end
                end
            end
            b = PaddedMatrices.PtrVector{$K,$T,$KL,$KL}(pointer(sp,$T))
            Base.Cartesian.@nexprs $K k -> b[k] = κ_k
        end
        #        ConstantFixedSizePaddedVector{$K,$T,$KL,$KL}( $outtup )'
        sp + $(VectorizationBase.align(KL*sizeof(T))), b'
    end
end

#=

@generated function Base.:*(
    C::AbstractMatrix{T},
    adj::Covariance_LAR_AR_Adjoint{K, nT, T, nTP}
) where {K, nT, T, nTP}
    # C is a DynamicCovarianceMatrix
    Wm1 = VectorizationBase.pick_vector_width(T)-1 
    KL = (K+Wm1) & ~Wm1
#    outtup = Expr(:tuple, [Expr(:call,:*,2,Symbol(:κ_,k)) for k ∈ 1:K]..., [zero(T) for k ∈ K+1:KL]...)
    quote
        $(Expr(:meta,:inline))
        # Calculate all in 1 pass?
        # ∂C∂ρ = MutableFixedSizePaddedVector{K,T}(undef)
        # Cstride = stride(C,2)
        L = adj.LKJ
        ∂ARs = adj.∂ARs
        Base.Cartesian.@nexprs $K j -> κ_j = zero($T)
        @inbounds begin
            Base.Cartesian.@nexprs $K kc -> begin
                # Diagonal block
#                kr = kc-1
                Base.Cartesian.@nexprs kc j -> l_j = L[kc,j]^2
                ccol = crow = $nT*(kc-1)
                for tc ∈ 1:$nT
                    tco = tc + ccol
                    for tr ∈ 1:tc-1
                        cij = C[tr + crow, tco]
                        Base.Cartesian.@nexprs kc j -> begin
                            κ_j = Base.FastMath.add_fast(Base.FastMath.mul_fast(cij,l_j,∂ARs[tr,tc,j]), κ_j)
                        end
                    end
                end

                for kr ∈ kc:$(K-1)
                    Base.Cartesian.@nexprs kc j -> l_j = L[kc,j]*L[kr+1,j]
                    ccol = $nT*kr; crow = $nT*(kc-1)
                    for tc ∈ 1:$nT
                        tco = tc + ccol
                        for tr ∈ 1:$nT
                            cij = C[tr + crow, tco]
                            Base.Cartesian.@nexprs kc j -> begin
                                κ_j = Base.FastMath.add_fast(Base.FastMath.mul_fast(cij,l_j,∂ARs[tr,tc,j]), κ_j)
                            end
                        end
                    end
                end
            end
        end
        (sp, out) = PtrVector{$K,$T}(sp)
        Base.Cartesian.@nexprs $K k -> out[k] = $(T(2))*κ_k
        #        ConstantFixedSizePaddedVector{$K,$T,$KL,$KL}( $outtup )
        sp, out'
    end
end

=#


function Base.:*(C::AbstractMatrix{T},
                    adj::Covariance_LAR_L_Adjoint{K, nT, T, nTP}
                ) where {K, nT, T, nTP}

    ∂LKJ = StructuredMatrices.MutableLowerTriangularMatrix{K,T}(undef)
    
        # $(Expr(:meta,:inline))

        # ∂LKJ = MutableLowerTriangualrMatrix{$K,$T}(undef)
        # ∂LKJ = $∂LKJ
    LKJ = adj.LKJ
    ARs = adj.ARs

    # Outer loops are for ∂lkj
    for kc ∈ 1:K
        # coffset = (kc-1)*stride(C,2)
        # AR[:,:,kc]
        @inbounds for kr ∈ kc:K
            ∂lkj = zero(T)
            # Move left
            for lr ∈ kc:kr-1
                lkj = LKJ[lr,kc]
                for tc ∈ 1:nT
                    # C: kr = row, lr = col
                    @fastmath for tr ∈ 1:nT
                        ∂lkj += lkj * ARs[tr,tc,kc] * C[tr + (kr-1)*nT,tc + (lr-1)*(nT)]
                    end
                end
            end
            # squared diagonal derivative
            lkj = 2LKJ[kr,kc]
            for tc ∈ 1:nT
                # C: kr = row, lr = col
#                @fastmath for tr ∈ 1:nT
#                    ∂lkj += lkj * ARs[tr,tc,kc] * C[tr + (kr-1)*nT,tc + (kr-1)*(nT)] #* (tc == tr ? T(0.5) : T(1.0))
#                end
                @fastmath ∂lkj += lkj * ARs[tc,tc,kc] * C[tc + (kr-1)*nT,tc + (kr-1)*(nT)]# * T(0.5)  #* (tc == tr ? T(0.5) : T(1.0))
                @fastmath for tr ∈ tc+1:nT
                    ∂lkj += lkj * ARs[tr,tc,kc] * C[tr + (kr-1)*nT,tc + (kr-1)*(nT)] #* (tc == tr ? T(0.5) : T(1.0))
                end
            end
            # move down
            for lr ∈ kr+1:K
                lkj = LKJ[lr,kc]
                for tc ∈ 1:nT
                    # C: lr = row, kr = col
                    @fastmath for tr ∈ 1:nT
                        ∂lkj += lkj * ARs[tr,tc,kc] * C[tr + (lr-1)*nT,tc + (kr-1)*(nT)]
                    end
                end
            end
            ∂LKJ[kr,kc] = ∂lkj
        end
    end
    ∂LKJ
end

function Base.:*(
    sp::StackPointer,
    C::AbstractMatrix{T},
    adj::Covariance_LAR_L_Adjoint{K, nT, T, nTP}
) where {K, nT, T, nTP}

    (sp,∂LKJ) = StructuredMatrices.PtrLowerTriangularMatrix{K,T}(sp)
    
        # $(Expr(:meta,:inline))

        # ∂LKJ = MutableLowerTriangualrMatrix{$K,$T}(undef)
        # ∂LKJ = $∂LKJ
    LKJ = adj.LKJ
    ARs = adj.ARs

    # Outer loops are for ∂lkj
    for kc ∈ 1:K
        # coffset = (kc-1)*stride(C,2)
        # AR[:,:,kc]
        @inbounds for kr ∈ kc:K
            ∂lkj = zero(T)
            # Move left
            for lr ∈ kc:kr-1
                lkj = LKJ[lr,kc]
                for tc ∈ 1:nT
                    # C: kr = row, lr = col
                    @fastmath for tr ∈ 1:nT
                        ∂lkj += lkj * ARs[tr,tc,kc] * C[tr + (kr-1)*nT,tc + (lr-1)*(nT)]
                    end
                end
            end
            # squared diagonal derivative
            lkj = 2LKJ[kr,kc]
            for tc ∈ 1:nT
                # C: kr = row, lr = col
#                @fastmath for tr ∈ 1:nT
#                    ∂lkj += lkj * ARs[tr,tc,kc] * C[tr + (kr-1)*nT,tc + (kr-1)*(nT)] #* (tc == tr ? T(0.5) : T(1.0))
#                end
                @fastmath ∂lkj += lkj * ARs[tc,tc,kc] * C[tc + (kr-1)*nT,tc + (kr-1)*(nT)] #* (tc == tr ? T(0.5) : T(1.0))
                @fastmath for tr ∈ tc+1:nT
                    ∂lkj += lkj * ARs[tr,tc,kc] * C[tr + (kr-1)*nT,tc + (kr-1)*(nT)] #* (tc == tr ? T(0.5) : T(1.0))
                end
            end
            # move down
            for lr ∈ kr+1:K
                lkj = LKJ[lr,kc]
                for tc ∈ 1:nT
                    # C: lr = row, kr = col
                    @fastmath for tr ∈ 1:nT
                        ∂lkj += lkj * ARs[tr,tc,kc] * C[tr + (lr-1)*nT,tc + (kr-1)*(nT)]
                    end
                end
            end
            ∂LKJ[kr,kc] = ∂lkj
        end
    end
#    @show ∂LKJ
    sp, ∂LKJ
end

