using PaddedMatrices: DynamicPaddedVector, DynamicPaddedMatrix, AbstractPaddedMatrix, AbstractDynamicPaddedMatrix
using PaddedMatrices: StackPointer, AbstractMutableFixedSizePaddedArray

struct DynamicCovarianceMatrix{T,ADT <: AbstractDynamicPaddedMatrix{T}} <: AbstractMatrix{T} #L
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
LinearAlgebra.checksquare(A::DynamicCovarianceMatrix) = nothing
Base.stride(A::DynamicCovarianceMatrix, i) = stide(A.data, i)


struct MissingDataVector{T,ADT,VI<:AbstractVector{Int}}#,VT<AbstractVector{T}}
    bitmask::BitVector
    indices::VI # note, indices start from 0
#    data::VT
    ∂Σ::DynamicCovarianceMatrix{T,ADT} # ∂Σ, rather than stack pointer, so that we can guarantee elements are 0
end
struct MissingDataVectorAdjoint{T,ADT,VI}#,VT}
    mdv::MissingDataVector{T,ADT,VI}#,VT}
end
function MissingDataVector{T}(bitmask::BitVector) where {T}
    N = length(bitmask)
#    data = zeros(DynamicPaddedMatrix{T}, (N,))
    Σ = DynamicCovarianceMatrix{T}(
        zeros(DynamicPaddedMatrix{T}, (N,N))
    )
    MissingDataVector(
        bitmask, findall(bitmask), Σ#data, Σ
    )
end
function MissingDataVector{T}(::Type{<:PaddedMatrices.AbstractFixedSizePaddedVector}, bitmask::BitVector) where {T}
    N = length(bitmask)
#    data = zeros(DynamicPaddedMatrix{T}, (N,))
    Σ = DynamicCovarianceMatrix{T}(
        zeros(DynamicPaddedMatrix{T}, (N,N))
    )
    indices = PaddedMatrices.MutableFixedSizePaddedVector(findall(bitmask))
    MissingDataVector(
        bitmask, indices, Σ#data, Σ
    )
end


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
    mask!(∂Σ, A, mdv.bitmask, mdv.indices)
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


function subset!(B::AbstractPaddedMatrix, A::AbstractPaddedMatrix, mdr::MissingDataVector, mdc::MissingDataVector)
    indsr = mdr.indices
    indsc = mdc.indices
    @boundscheck begin
        size(B) == (length(indsr), length(indsc)) || PaddedMatrices.ThrowBoundsError()
        size(A) == (length(mdr.bitmask), length(mdc.bitmask)) || PaddedMatrices.ThrowBoundsError()
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
        size(A) == (length(mdv.bitmask), length(mdv.bitmask)) || PaddedMatrices.ThrowBoundsError()
    end

    @inbounds for ics ∈ eachindex(inds), irs ∈ ics:N #eachindex(indsr)
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
    B, MissingDataVector(mdv)
end
function PaddedMatrices.∂getindex( sp::StackPointer, A::DynamicCovarianceMatrix{T}, mdv::MissingDataVector ) where {T}
    sp, B = DynamicCovarianceMatrix{T}( sp, length(mdv.indices) )
    @inbounds subset!( B, A, mdv )
    sp, (B, MissingDataVectorAdjoint(mdv))
end
function Base.getindex(
    a::AbstractMutableFixedSizePaddedVector{P,T},
    mdv::MissingDataVector{T,ADT,V}
) where {T,ADT,P,V<:PaddedMatrices.AbstractFixedSizePaddedVector{P,Int}}
    inds = mdv.indices
    b = MutableFixedSizePaddedVector{P,T}(undef)
    @inbounds for p ∈ 1:P
        b[p] = a[inds[p]]
    end
    b
end
function PaddedMatrices.∂getindex(
    a::AbstractMutableFixedSizePaddedVector{P,T},
    mdv::MissingDataVector{T,ADT,V}
) where {T,ADT,P,V<:PaddedMatrices.AbstractFixedSizePaddedVector{P,Int}}
    inds = mdv.indices
    b = MutableFixedSizePaddedVector{P,T}(undef)
    @inbounds for p ∈ 1:P
        b[p] = a[inds[p]]
    end
    b, MissingDataVectorAdjoint(mdv)
end
function Base.getindex(
    sp::StackPointer,
    a::AbstractMutableFixedSizePaddedVector{P,T},
    mdv::MissingDataVector{T,ADT,V}
) where {T,ADT,P,V<:PaddedMatrices.AbstractFixedSizePaddedVector{P,Int}}
    inds = mdv.indices
    sp, b = PaddedMatrices.PtrVector{P,T}(sp)
    @inbounds for p ∈ 1:P
        b[p] = a[inds[p]]
    end
    sp, b
end
function PaddedMatrices.∂getindex(
    sp::StackPointer,
    a::AbstractMutableFixedSizePaddedVector{P,T},
    mdv::MissingDataVector{T,ADT,V}
) where {T,ADT,P,V<:PaddedMatrices.AbstractFixedSizePaddedVector{P,Int}}
    inds = mdv.indices
    sp, b = PaddedMatrices.PtrVector{P,T}(sp)
    @inbounds for p ∈ 1:P
        b[p] = a[inds[p]]
    end
    sp, (b, MissingDataVectorAdjoint(mdv))
end


function Base.:*(mdv::MissingDataVectorAdjoint, A::DynamicCovarianceMatrix)
    ∂Σ = mdv.∂Σ
    mask!(∂Σ, A, mdv.bitmask, mdv.indices)
    ∂Σ
end
function Base.:*(mdv::MissingDataVectorAdjoint, A::AbstractVector)
    b = DynamicPaddedVector
end
@generated function Base.:*(
    mdv::MissingDataVectorAdjoint,
    a::NTuple{K,V}
) where {K,P,T,V<:PaddedMatrices.AbstractMutableFixedSizePaddedVector{P,T}}
    quote
        Base.Cartesian.@nexprs $K k -> (b_k = zeros(MutableFixedSizePaddedVector{$P,$T}); a_k = a[k])
        inds = mdv.indices
        @inbounds for p ∈ 1:$P
            Base.Cartesian.@nexprs $K k -> b_k[inds[p]] = a_k[i]
        end
        Base.Cartesian.@ntuple $K b
    end
end
@generated function Base.:*(
    sp::StackPointer,
    mdv::MissingDataVectorAdjoint,
    a::NTuple{K,V}
) where {K,P,T,V<:PaddedMatrices.AbstractMutableFixedSizePaddedVector{P,T}}
    Wm1 = VectorizationBase.pick_vector_width(P,T) - 1
    padded_length = (P + Wm1) & ~Wm1
    total_length = K*padded_length
    quote
        # we zero the used data in a single loop.
        zero_out = PtrVector{$total_length,$T}(pointer(sp, $T))
        @inbounds @simd ivdep for i ∈ 1:$total_length
            zero_out[i] = zero($T)
        end
        
        Base.Cartesian.@nexprs $K k -> begin
            sp, b_k = PtrVector{$P,$T}(sp)
            a_k = a[k]
        end
        inds = mdv.indices
        @inbounds for p ∈ 1:$P
            Base.Cartesian.@nexprs $K k -> b_k[inds[p]] = a_k[i]
        end
        sp, (Base.Cartesian.@ntuple $K b)
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
        Sigfull
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
        Sigfull, Covariance_LAR_AR_Adjoint(∂ARs,L), Covariance_LAR_L_Adjoint(ARs,L)
    end
end

@generated function DynamicCovarianceMatrix(
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
    quote
        # only Sigfull escapes, so we allocate it first
        # and return the stack pointer pointing to its end.
        sp, Sigfull = PaddedMatrices.PtrMatrix{$KT,$KT,$T,$KTR}(sp)
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
                for tc ∈ 1:$nT
                    @simd ivdep for tr ∈ 1:$nTl
                        ari = l_1 * ARs[tr, tc, 1]
                        Base.Cartesian.@nexprs kc-1 j -> ari = muladd(l_{j+1}, ARs[tr, tc, j+1], ari)
                        Sigfull[ tr + sigrow, tc + sigcol ] = ari
                    end
                end
            end
        end
        end
#        sp, (Sigfull,(∂ARs,L), (ARs,L))
        sp, Sigfull
    end
end
@generated function ∂DynamicCovarianceMatrix(
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
              end : quote
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
              end )
        end
        sp, Sigfull = PaddedMatrices.PtrMatrix{$KT,$KT,$T,$KTR}(sp)
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
#        sp, (Sigfull,(∂ARs,L), (ARs,L))
        sp, (Sigfull, Covariance_LAR_AR_Adjoint(∂ARs,L), Covariance_LAR_L_Adjoint(ARs,L))
    end
end
PaddedMatrices.@support_stack_pointer DistributionParameters DynamicCovarianceMatrix
PaddedMatrices.@support_stack_pointer DistributionParameters ∂DynamicCovarianceMatrix

@generated function Base.:*(C::AbstractMatrix{T},
                adj::Covariance_LAR_AR_Adjoint{K, nT, T, nTP}
                ) where {K, nT, T, nTP}
    # C is a DynamicCovarianceMatrix
    Wm1 = VectorizationBase.pick_vector_width(T)-1 
    KL = (K+Wm1) & ~Wm1
    outtup = Expr(:tuple, [Expr(:call,:*,2,Symbol(:κ_,k)) for k ∈ 1:K]..., [zero(T) for k ∈ K+1:KL]...)
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
        ConstantFixedSizePaddedVector{$K,$T,$KL,$KL}( $outtup )
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
            # squared diagonal
            lkj = LKJ[kr,kc]
            for tc ∈ 1:nT
                # C: kr = row, lr = col
#                @fastmath for tr ∈ 1:nT
#                    ∂lkj += lkj * ARs[tr,tc,kc] * C[tr + (kr-1)*nT,tc + (kr-1)*(nT)] #* (tc == tr ? T(0.5) : T(1.0))
#                end
                @fastmath ∂lkj += lkj * ARs[tc,tc,kc] * C[tc + (kr-1)*nT,tc + (kr-1)*(nT)] #* (tc == tr ? T(0.5) : T(1.0))
                @fastmath for tr ∈ tc+1:nT
                    ∂lkj += 2lkj * ARs[tr,tc,kc] * C[tr + (kr-1)*nT,tc + (kr-1)*(nT)] #* (tc == tr ? T(0.5) : T(1.0))
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
            ∂LKJ[kr,kc] = 2∂lkj
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
            # squared diagonal
            lkj = LKJ[kr,kc]
            for tc ∈ 1:nT
                # C: kr = row, lr = col
#                @fastmath for tr ∈ 1:nT
#                    ∂lkj += lkj * ARs[tr,tc,kc] * C[tr + (kr-1)*nT,tc + (kr-1)*(nT)] #* (tc == tr ? T(0.5) : T(1.0))
#                end
                @fastmath ∂lkj += lkj * ARs[tc,tc,kc] * C[tc + (kr-1)*nT,tc + (kr-1)*(nT)] #* (tc == tr ? T(0.5) : T(1.0))
                @fastmath for tr ∈ tc+1:nT
                    ∂lkj += 2lkj * ARs[tr,tc,kc] * C[tr + (kr-1)*nT,tc + (kr-1)*(nT)] #* (tc == tr ? T(0.5) : T(1.0))
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
            ∂LKJ[kr,kc] = 2∂lkj
        end
    end
    sp, ∂LKJ
end

