using PaddedMatrices: DynamicPaddedVector, DynamicPaddedMatrix, AbstractPaddedMatrix, AbstractDynamicPaddedMatrix
using PaddedMatrices: StackPointer, AbstractMutableFixedSizeArray

abstract type AbstractFixedSizeCovarianceMatrix{M,T,R,L} <: PaddedMatrices.AbstractMutableFixedSizeMatrix{M,M,T,R,L} end
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
    R = M
    L = R * M
    :(sp + $(sizeof(T)*L), PtrFixedSizeCovarianceMatrix{$M,$T,$R,$L}(pointer(sp, $T)))
end
@generated function MutableFixedSizeCovarianceMatrix{M,T}(::UndefInitializer) where {M,T}
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

struct MissingDataVector{T,S,M,R,L}
    indices::MutableFixedSizeVector{S,Int,S} # note, indices start from 0
    ∂Σ::MutableFixedSizeCovarianceMatrix{M,T,R,L} # ∂Σ, rather than stack pointer, so that we can guarantee elements are 0
end
struct MissingDataVectorAdjoint{T,S,M,R,L}#,VT}
    mdv::MissingDataVector{T,S,M,R,L}#,VT}
end
function MissingDataVector{T}(bitmask) where {T}
    MissingDataVector{T}(PaddedMatrices.MutableFixedSizeVector, bitmask)
end
## Not Type Stable
function MissingDataVector{T}(::Type{<:PaddedMatrices.AbstractFixedSizeVector}, bitmask) where {T}
    Σ = zeros(MutableFixedSizeCovarianceMatrix{length(bitmask),T})
    inds = findall(i -> i == 1, bitmask)
    indices = PaddedMatrices.MutableFixedSizeVector{length(inds),Int}(inds)#::PaddedMatrices.MutableFixedSizeVector
    MissingDataVector(
        indices, Σ
    )
end
Base.size(::MissingDataVectorAdjoint{T,S,M}) where {T,S,M} = (S,M)

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
    inds = mdv.indices
    N = length(inds)
    @boundscheck begin
        size(B) == (N, N) || PaddedMatrices.ThrowBoundsError()
    end
    @inbounds for ics ∈ eachindex(inds), irs ∈ ics:N #eachindex(indsr)
        B[irs,ics] = A[inds[irs],inds[ics]]
    end
end
function subset!(
    B::AbstractFixedSizeCovarianceMatrix{S,T},
    A::AbstractFixedSizeCovarianceMatrix{L,T},
    mdv::MissingDataVector{T,S,L}
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
    a::AbstractMutableFixedSizeVector{M,T},
    mdv::MissingDataVector{T,S,M}
) where {T,S,M}
    inds = mdv.indices
    b = MutableFixedSizeVector{S,T}(undef)
    @inbounds for s ∈ 1:S
        b[s] = a[inds[s]]
    end
    b
end
function PaddedMatrices.∂getindex(
    a::AbstractMutableFixedSizeVector{M,T},
    mdv::MissingDataVector{T,S,M}
) where {T,S,M}
    inds = mdv.indices
    b = MutableFixedSizeVector{S,T}(undef)
    @inbounds for s ∈ 1:S
        b[s] = a[inds[s]]
    end
    b, MissingDataVectorAdjoint(mdv)
end
function Base.getindex(
    sp::StackPointer,
    a::AbstractMutableFixedSizeVector{M,T},
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
    a::AbstractMutableFixedSizeVector{M,T},
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
) where {T,K,S,M,V<:PaddedMatrices.AbstractMutableFixedSizeVector{M,T}}
    quote
        inds = mdv.indices
        Base.Cartesian.@nexprs $K k -> (a_k = @inbounds a[k]; (sp, b_k) = PaddedMatrices.PtrVector{$S,$T}(sp))
        for s ∈ 1:$S
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
) where {T,K,S,M,V<:PaddedMatrices.AbstractMutableFixedSizeVector{M,T}}
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
) where {K,T,S,M,V<:PaddedMatrices.AbstractMutableFixedSizeVector{S,T}}
    quote
        Base.Cartesian.@nexprs $K k -> (b_k = zeros(MutableFixedSizeVector{$M,$T}); a_k = a[k])
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
) where {K,T,S,M,V<:PaddedMatrices.AbstractMutableFixedSizeVector{S,T}}
    Wm1 = VectorizationBase.pick_vector_width(M,T) - 1
    total_length = (K*M + Wm1) & + ~Wm1
    quote
        # we zero the used data in a single loop.
        zero_out = PtrVector{$total_length,$T,$total_length}(pointer(sp, $T))
        @inbounds @simd ivdep for i ∈ 1:$total_length
            zero_out[i] = zero($T)
        end
        Base.Cartesian.@nexprs $K k -> begin
            b_k = PtrVector{$M,$T,$M}(pointer(sp,$T) + $(sizeof(T)*M) * (k-1))
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

@generated function DynamicCovarianceMatrix(
                rhos::PaddedMatrices.AbstractFixedSizeVector{K,T}, L::StructuredMatrices.AbstractLowerTriangularMatrix{K},
                times::ConstantFixedSizeVector{nT}, workspace
            ) where {K,T,nT}
    Wm1 = VectorizationBase.pick_vector_width(nT, T) - 1
    nTl = (nT + Wm1) & ~Wm1
    quote
        ARs = workspace.ARs
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
        end
        Sigfull = workspace.Sigfull
        @inbounds begin
        Base.Cartesian.@nexprs $K kc -> begin
            for kr ∈ kc-1:K-1
                Base.Cartesian.@nexprs kc j -> l_j = L[kr+1,j] * L[kc,j]
                sigrow, sigcol = $nT*kr, $nT*(kc-1)
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


struct Covariance_LAR_AR_Adjoint{K, nT, T, nTP, L, MFPD<:AbstractMutableFixedSizeArray{Tuple{nT,nT,K},T,3,nTP,L}, LKJ_T <: StructuredMatrices.AbstractLowerTriangularMatrix}
    ∂ARs::MFPD#MutableFixedSizeArray{Tuple{nT,nT,K},T,3,nTP,L}
    LKJ::LKJ_T
end
struct Covariance_LAR_L_Adjoint{K, nT, T, nTP, L, MFPD<:AbstractMutableFixedSizeArray{Tuple{nT,nT,K},T,3,nTP,L}, LKJ_T <: StructuredMatrices.AbstractLowerTriangularMatrix}
    ARs::MFPD#MutableFixedSizeArray{Tuple{nT,nT,K},T,3,nTP,L}
    LKJ::LKJ_T
end

@generated function ∂DynamicCovarianceMatrix(
    rhos::PaddedMatrices.AbstractFixedSizeVector{K,T},
    L::StructuredMatrices.AbstractLowerTriangularMatrix{K,T},
    times::AbstractFixedSizeVector{nT},
    workspace, ::Val{(true,true)}
) where {K,T,nT}
    # We will assume rho > 0
    Wm1 = VectorizationBase.pick_vector_width(nT, T) - 1
    nTl = (nT + Wm1) & ~Wm1
    quote
        # nT = length(times) + 1
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
                    # We want to specifically use Base.exp, and not fastmath exp
                    logrhot = (deltatimes - one($T))*logρ
                    rhot = $(T == Float64 ? :(ccall(:exp,Float64,(Float64,),logrhot)) : :(Base.exp(logrhot)))
                    ARs[tr,tc,k] = ρ*rhot
                    ∂ARs[tr,tc,k] = deltatimes*rhot
                end
            end
        end
        Sigfull = workspace.Sigfull
        @inbounds begin
        Base.Cartesian.@nexprs $K kc -> begin
            for kr ∈ kc-1:K-1
                Base.Cartesian.@nexprs kc j -> l_j = L[kr+1,j] * L[kc,j]
                sigrow, sigcol = $nT*kr, $nT*(kc-1)
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
    rhos::PaddedMatrices.AbstractFixedSizeVector{K,T},
    L::StructuredMatrices.AbstractLowerTriangularMatrix{K,T},
    times::AbstractFixedSizeVector{nT},
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
            for tcouter ∈ 0:$((nTl>>>Wshift)-1)
                for tcinner ∈ 1:$W
                    Wtcouter = $W*tcouter
                    tc = tcinner + Wtcouter
                    tc > $nT && break
                    for tr ∈ 1:Wtcouter
                        ARs[tr,tc,k] = ARs[tc,tr,k]
                    end
                    vtime_tc = SIMDPirates.vbroadcast($V, times[tc])
                    for i ∈ tcouter:$((nTl>>>Wshift)-1)
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
        @inbounds begin
            Base.Cartesian.@nexprs $K kc -> begin
                for kr ∈ kc-1:K-1
                    Base.Cartesian.@nexprs kc j -> l_j = L[kr+1,j] * L[kc,j]
                    sigrow, sigcol = $nT*kr, $nT*(kc-1)
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
        sp, Sigfull
    end
end
@generated function ∂CovarianceMatrix(
    sp::StackPointer,
    rhos::PaddedMatrices.AbstractFixedSizeVector{K,T},
    L::StructuredMatrices.AbstractLowerTriangularMatrix{K,T},
    times::AbstractFixedSizeVector{nT},
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
    sym = true
    ARquote = if SIMD && sym
        quote
            vρ = SIMDPirates.vbroadcast($V, ρ)
            vlogρ = SIMDPirates.vbroadcast($V, logρ)
            for tcouter ∈ 0:$((nTl>>>Wshift)-1)
                for tcinner ∈ 1:$W
                    Wtcouter = $W*tcouter
                    tc = tcinner + Wtcouter
                    tc > $nT && break
                    for tr ∈ 1:Wtcouter
                        ARs[tr,tc,k] = ARs[tc,tr,k]
                        ∂ARs[tr,tc,k] = ∂ARs[tc,tr,k]
                    end
                    vtime_tc = SIMDPirates.vbroadcast($V, times[tc])
                    for i ∈ tcouter:$((nTl>>>Wshift)-1)
                        vtimes_tr = SIMDPirates.vload($V, ptr_time + i * $(T_size*W))
                        vδt = SIMDPirates.vabs(SIMDPirates.vsub(vtimes_tr, vtime_tc))
                        vδtm1 = SIMDPirates.vsub(vδt, SIMDPirates.vbroadcast($V,one($T)))
                        vρt = SLEEFPirates.exp(SIMDPirates.vmul(vδtm1, vlogρ))
                        AR_offset = (k-1)*$(T_size*nTl*nT) + (tc-1)*$(T_size*nTl) + i*$(W*T_size)

                        SIMDPirates.vstore!(ptr_ARs + AR_offset, SIMDPirates.vmul(vρ,vρt))
                        SIMDPirates.vstore!(ptr_∂ARs + AR_offset, SIMDPirates.vmul(vδt,vρt))
                    end
                end
              end
        end
    elseif SIMD
        quote
            vρ = SIMDPirates.vbroadcast($V, ρ)
            vlogρ = SIMDPirates.vbroadcast($V, logρ)
            @inbounds for c ∈ 0:$(nT-1)
                offset = (k-1)*$(T_size*nTl*nT) + c*$(T_size*nTl)
                vtime_tc = SIMDPirates.vbroadcast($V, times[c+1])
                for r ∈ 0:$((nTl >>> Wshift)-1)
                    vtimes_tr = SIMDPirates.vload($V, ptr_time + r * $(T_size*W))
                    vδt = SIMDPirates.vabs(SIMDPirates.vsub(vtimes_tr, vtime_tc))
                    vδtm1 = SIMDPirates.vsub(vδt, SIMDPirates.vbroadcast($V,one($T)))
                    vρt = SLEEFPirates.exp(SIMDPirates.vmul(vδtm1, vlogρ))
                    AR_offset = offset + r*$(W*T_size)

                    SIMDPirates.vstore!(ptr_ARs + AR_offset, SIMDPirates.vmul(vρ,vρt))
                    SIMDPirates.vstore!(ptr_∂ARs + AR_offset, SIMDPirates.vmul(vδt,vρt))
                end
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
        sp,  ARs = PaddedMatrices.PtrArray{Tuple{$nT,$nT,$K},$T,3,$nTl}(sp)
        sp, ∂ARs = PaddedMatrices.PtrArray{Tuple{$nT,$nT,$K},$T,3,$nTl}(sp)
        ptr_time = pointer(times)
        ptr_ARs = pointer(ARs)
        ptr_∂ARs = pointer(∂ARs)
        @inbounds for k ∈ 1:$K
            ρ = rhos[k]
            # We want to use Base.log, and not fastmath log.
            logρ = $(T == Float64 ? :(ccall(:log,Float64,(Float64,),ρ))  :  :(Base.log(ρ)))
            $ARquote
        end
        sp, Sigfull = PtrFixedSizeCovarianceMatrix{$KT,$T}(sp)
        @inbounds begin
            Base.Cartesian.@nexprs $K kc -> begin
                for kr ∈ kc-1:K-1
                    Base.Cartesian.@nexprs kc j -> l_j = L[kr+1,j] * L[kc,j]
                    sigrow, sigcol = $nT*kr, $nT*(kc-1)
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
        sp, (Sigfull, Covariance_LAR_AR_Adjoint(∂ARs,L), Covariance_LAR_L_Adjoint(ARs,L))
    end
end

@generated function Base.:*(
    C::AbstractMatrix{T},
    adj::Covariance_LAR_AR_Adjoint{K, nT, T, nTP}
) where {K, nT, T, nTP}
    # C is a DynamicCovarianceMatrix
    Wm1 = VectorizationBase.pick_vector_width(T)-1 
    KL = (K+Wm1) & ~Wm1
    outtup = Expr(:tuple, [Symbol(:κ_,k) for k ∈ 1:K]..., [zero(T) for k ∈ K+1:KL]...)
    quote
        $(Expr(:meta,:inline))
        # Calculate all in 1 pass?
        L = adj.LKJ
        ∂ARs = adj.∂ARs
        Base.Cartesian.@nexprs $K j -> κ_j = zero($T)
        @inbounds begin
            Base.Cartesian.@nexprs $K kc -> begin
                # Diagonal block
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
        ConstantFixedSizeVector{$K,$T,$KL,$KL}( $outtup )'
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
    quote
        $(Expr(:meta,:inline))
        # Calculate all in 1 pass?
        L = adj.LKJ
        ∂ARs = adj.∂ARs
        Base.Cartesian.@nexprs $K j -> κ_j = zero($T)
        @inbounds begin
            Base.Cartesian.@nexprs $K kc -> begin
                # Diagonal block
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
            b = PaddedMatrices.PtrVector{$K,$T,$KL}(pointer(sp,$T))
            Base.Cartesian.@nexprs $K k -> b[k] = κ_k
        end
        sp + $(VectorizationBase.align(KL*sizeof(T))), b'
    end
end

function Base.:*(
    C::AbstractMatrix{T},
    adj::Covariance_LAR_L_Adjoint{K, nT, T, nTP}
) where {K, nT, T, nTP}
    ∂LKJ = StructuredMatrices.MutableLowerTriangularMatrix{K,T}(undef)
    LKJ = adj.LKJ
    ARs = adj.ARs
    # Outer loops are for ∂lkj
    for kc ∈ 1:K
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
    
    LKJ = adj.LKJ
    ARs = adj.ARs

    # Outer loops are for ∂lkj
    for kc ∈ 1:K
        # coffset = (kc-1)*stride(C,2)
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

