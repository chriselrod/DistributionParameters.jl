using PaddedMatrices: PaddedMatrix

struct ∂MultivariateNormalVariate{T} <: AbstractMatrix{T}
    data::PaddedMatrix{T}
end
Base.size(A::∂MultivariateNormalVariate) = size(A.data)
Base.getindex(A::∂MultivariateNormalVariate, I...) = Base.getindex(A.data, I...)

struct MultivariateNormalVariate{T} <: AbstractMatrix{T}
    data::PaddedMatrix{T}
    δ::PaddedMatrix{T}
    Σ⁻¹δ::∂MultivariateNormalVariate{T}
end
Base.size(A::MultivariateNormalVariate) = size(A.data)
Base.getindex(A::MultivariateNormalVariate, I...) = Base.getindex(A.data, I...)

struct CovarianceMatrix{T} <: AbstractMatrix{T} #L
    Σ::Symmetric{T,PaddedMatrix{T}}
    ∂Σ::PaddedMatrix{T}
end
Base.size(A::CovarianceMatrix) = size(A.data)
Base.getindex(A::CovarianceMatrix, I...) = Base.getindex(A.data, I...)

function LinearAlgebra.cholesky!(Σ::CovarianceMatrix)
    # LinearAlgebra.LAPACK.potrf!('U', Σ.Data)
    cholesky!(Σ.Σ)
end

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



@generated function CovarianceMatrix(
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
            rho = rhos[k]
            for tc ∈ 1:nT
                for tr ∈ 1:tc-1
                    ARs[tr,tc,k] = ARs[tc,tr,k]
                end
                ARs[tc,tc,k] = one(T)
                for tr ∈ tc+1:nT
                    deltatimes = times[tr] - times[tc]
                    rhot = rho^(deltatimes - one(T))
                    ARs[tr,tc,k] = rho*rhot
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
                sigrow, sigcol = $nT*(kc-1), $nT*kr # transpose it
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

struct Covariance_LAR_AR_Adjoint{K, nT, T, nTP, L, LKJ_T <: StructuredMatrices.AbstractLowerTriangularMatrix}
    ∂ARs::MutableFixedSizePaddedArray{Tuple{nT,nT,K},T,3,nTP,L}
    LKJ::LKJ_T
end
struct Covariance_LAR_L_Adjoint{K, nT, T, nTP, L, LKJ_T <: StructuredMatrices.AbstractLowerTriangularMatrix}
    ARs::MutableFixedSizePaddedArray{Tuple{nT,nT,K},T,3,nTP,L}
    LKJ::LKJ_T
end

@generated function ∂CovarianceMatrix(
                rhos::PaddedMatrices.AbstractFixedSizePaddedVector{K,T}, L::StructuredMatrices.AbstractLowerTriangularMatrix{K,T},
                times::ConstantFixedSizePaddedVector{nT}, workspace, ::Val{(true,true)}
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
            rho = rhos[k]
            for tc ∈ 1:nT
                for tr ∈ 1:tc-1
                    ARs[tr,tc,k] = ARs[tc,tr,k]
                    ∂ARs[tr,tc,k] = ∂ARs[tc,tr,k]
                end
                ARs[tc,tc,k] = one(T)
                ∂ARs[tc,tc,k] = zero(T)
                for tr ∈ tc+1:nT
                    deltatimes = times[tr] - times[tc]
                    rhot = rho^(deltatimes - one(T))
                    ARs[tr,tc,k] = rho*rhot
                    ∂ARs[tr,tc,k] = deltatimes*rhot
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
                sigrow, sigcol = $nT*(kc-1), $nT*kr # transpose it
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


@generated function Base.:*(C::AbstractMatrix{T},
                adj::Covariance_LAR_AR_Adjoint{K, nT, T, nTP}
                ) where {K, nT, T, nTP}
    # C is a CovarianceMatrix
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
 
# function calc_lkj_partial(ar, ::Val{K}, ::Val{KC}) where {K, KC}

# end

# @generated function Base.:*(C::AbstractMatrix{T},
#                     adj::Covariance_LAR_L_Adjoint{K, nT, T, nTP}
#                 ) where {K, nT, T, nTP}

#     ∂LKJ = StructuredMatrices.MutableLowerTriangularMatrix{K,T}(undef)
#     quote
#         # $(Expr(:meta,:inline))

#         # ∂LKJ = MutableLowerTriangualrMatrix{$K,$T}(undef)
#         ∂LKJ = $∂LKJ
#         LKJ = adj.LKJ
#         ARs = adj.ARs

#         for kc ∈ 1:$K
#             # coffset = (kc-1)*stride(C,2)
#             # AR[:,:,kc]
#             @inbounds for kr ∈ kc:$K
#                 ∂lkj = zero(T)
#                 for lr ∈ kc:kr-1
#                     lkj = LKJ[lr,kc]
#                     for tc ∈ 1:$nT
#                         # C: kr = row, lr = col
#                         @fastmath for tr ∈ 1:$nT
#                             ∂lkj += lkj * ARs[tr,tc,kc] * C[tr + (kr-1)*$nT,tc + (lr-1)*$(K*nT)]
#                         end
#                     end
#                 end
#                 lkj = 2LKJ[kr,kc]
#                 for tc ∈ 1:$nT
#                     # C: kr = row, lr = col
#                     @fastmath for tr ∈ 1:$nT
#                         ∂lkj += lkj * ARs[tr,tc,kc] * C[tr + (kr-1)*$nT,tc + (kr-1)*$(K*nT)]
#                     end
#                 end
#                 for lr ∈ kr+1:$K
#                     lkj = LKJ[lr,kc]
#                     for tc ∈ 1:$nT
#                         # C: lr = row, kr = col
#                         @fastmath for tr ∈ 1:$nT
#                             ∂lkj += lkj * ARs[tr,tc,kc] * C[tr + (kr-1)*$nT,tc + (kr-1)*$(K*nT)]
#                         end
#                     end
#                 end
#                 ∂LKJ[kr,kc] = ∂lkj
#             end
#         end
#         ∂LKJ
#     end
# end

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

