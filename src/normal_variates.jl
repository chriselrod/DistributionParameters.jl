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

function Base.(:\)(U::UpperTriangular{T,Matrix{T}}, Y::MultivariateNormalVariate{T}) where {T}
    δ = Y.δ
    # Σ⁻¹δ = Y.Σ⁻¹δ
    # copyto!(Σ⁻¹δ, δ)
    LinearAlgebra.LAPACK.trtrs!('L', 'N', 'N', U.data, δ)
end

function Base.(:\)(U::Cholesky{T,Matrix{T}}, Y::MultivariateNormalVariate{T}) where {T}
    δ = Y.δ
    Σ⁻¹δ = Y.Σ⁻¹δ.data
    copyto!(Σ⁻¹δ, δ)
    LinearAlgebra.LAPACK.potrs!('L', U.factors, Σ⁻¹δ)
end



function CovarianceMatrix(
                rhos, L::StructuredMatrices.AbstractLowerTriangularMatrix{K,T},
                δₜ::ConstantFixedSizePaddedVector{nT}, workspace
            ) where {K,T,nT}
    Wm1 = VectorizationBase.pick_vector_width(nT, T) - 1
    nTl = (nT + Wm1) & ~Wm1
    quote
        nT = length(δₜ) + 1
        # K = size(L,1)
        # ARs = workspace.ARs
        Base.Cartesian.@nexprs $K k -> AR_k = ConstantFixedSizePaddedMatrix(AutoregressiveMatrix(rhos[k], δₜ))
        # ARs = [ConstantFixedSizePaddedMatrix(AutoregressiveMatrix(rho, δₜ)) for rho ∈ rhos]
        Sigfull = workspace.Sigfull
        # LowerTriangular
        Base.Cartesian.@nexprs $K kc -> begin
            for kr ∈ kc-1:K-1
                Base.Cartesian.@nexprs kc j -> l_k = L[kr+1,j] * L[kc,j]
                # Sigview = @view Sigfull[(1:$nTl) .+ $nT*kr, (1:$nT) .+ $nT*(kc-1)]
                sigrow, sigcol = $nT*kr, $nT*(kc-1)
                for tc ∈ 1:$nT
                    @simd ivdep for tr ∈ 1:$nTl
                        ari = l_1 * ARs_1[tr, tc]
                        Base.Cartesian.@nexprs kc j -> ari = muladd(l_{j+1}, AR_{j+1}[tr, tc], ari)
                        Sigfull[ tr + sigrow, tc + sigcol] = ari
                    end
                end
            end
        end
        Sigfull
    end
end
function ∂CovarianceMatrix(
                rhos::PaddedMatrices.AbstractFixedSizePaddedVector{K,T}, L::StructuredMatrices.AbstractLowerTriangularMatrix{K,T},
                times::ConstantFixedSizePaddedVector{nT}, workspace, ::Val{(true,true)}
            ) where {K,T,nT}
    Wm1 = VectorizationBase.pick_vector_width(nT, T) - 1
    nTl = (nT + Wm1) & ~Wm1
    quote
        # nT = length(times) + 1
        # K = size(L,1)
        ARs = workspace.ARs
        ∂ARs = workspace.∂ARs
        for k ∈ 1:k
            rho = rhos[k]
            if rho < 0
                absrho = abs(rho)
                for tc ∈ 1:nT
                    for tr ∈ 1:tc-1
                        sign = (-1)^(tc-tr)
                        deltatimes = times[tc] - times[tr]
                        rhot = sign* absrho^(deltatimes - one(T))
                        ARs[tr,tc,k] = rho*rhot
                        ∂ARs[tr,tc,k] = deltatimes*rhot
                    end
                    ARs[tc,tc,k] = one(T)
                    ∂ARs[tc,tc,k] = zero(T)
                end
            else # rho > 0
                for tc ∈ 1:nT
                    for tr ∈ 1:tc-1
                        deltatimes = times[tc] - times[tr]
                        rhot = rho^(deltatimes - one(T))
                        ARs[tr,tc,k] = rho*rhot
                        ∂ARs[tr,tc,k] = deltatimes*rhot
                    end
                    ARs[tc,tc,k] = one(T)
                    ∂ARs[tc,tc,k] = zero(T)
                end
            end
            ARs[:,:,k] .= AutoregressiveMatrix(rhos[k], δₜ)
        end
        # Base.Cartesian.@nexprs $K k -> AR_k = ConstantFixedSizePaddedMatrix(AutoregressiveMatrix(rhos[k], δₜ))
        # ARs = [ConstantFixedSizePaddedMatrix(AutoregressiveMatrix(rho, δₜ)) for rho ∈ rhos]
        Sigfull = workspace.Sigfull
        # ∂Sig∂L
        Base.Cartesian.@nexprs $K kc -> begin
            for kr ∈ kc-1:K-1
                Base.Cartesian.@nexprs kc j -> l_k = L[kr+1,j] * L[kc,j]
                # Sigview = @view Sigfull[(1:$nTl) .+ $nT*kr, (1:$nT) .+ $nT*(kc-1)]
                sigrow, sigcol = $nT*(kc-1), $nT*kr # transpose it
                for tc ∈ 1:$nT
                    @simd ivdep for tr ∈ 1:$nTl
                        ari = l_1 * ARs_1[tr, tc]
                        Base.Cartesian.@nexprs kc j -> ari = muladd(l_{j+1}, AR_{j+1}[tr, tc], ari)
                        Sigfull[ tr + sigrow, tc + sigcol] = ari
                    end
                end
            end
        end
        Sigfull
    end
end

