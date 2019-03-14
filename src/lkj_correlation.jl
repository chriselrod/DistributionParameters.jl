
struct LKJ_Correlation_Cholesky{M,T,L} <: AbstractMatrix{T}
    data::NTuple{T,L}
    # inverse::NTuple{T,L}
end
struct Inverse_LKJ_Correlation_Cholesky{M,T,L} <: AbstractMatrix{T}
    data::NTuple{T,L}
    # inverse::NTuple{T,L}
end

function load_parameter(first_pass, second_pass, out, ::Type{<: LKJ_Correlation_Cholesky{M,T}}, partial = false) where {M,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    ninvlogitout = gensym(out)
    invlogitout = gensym(out)
    ∂invlogitout = gensym(out)

    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    sumθᵢ = gensym(:sumθᵢ)

    N = (M * (M-1)) >> 1

    Wm1 = W - 1
    rem = N & Wm1
    L = (N + Wm1) & ~Wm1

    log_jac = gensym(:log_jac)
    q = quote
        $mv = MutableFixedSizePaddedVector{$M,$T}(undef)
        $log_jac = DistributionParameters.SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T))
    end
    if partial
        push!(q.args, :($invlogitout = MutableFixedSizePaddedVector{$M,$T}(undef)))
        push!(q.args, :($∂invlogitout = MutableFixedSizePaddedVector{$M,$T}(undef)))
    end
    loop_body = quote
        $ninvlogitout = one($T) / (one($T) + SLEEFPirates.exp($θ[i]))
    end
    if partial
        push!(loop_body.args, :($invlogitout[i] = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout[i] = $ninvlogitout * $invlogitout[i]))
        push!(loop_body.args, :($mv[i] = SIMDPirates.vmuladd($(T(2)), $invlogitout, one($T))))
        push!(loop_body.args, :($log_jac += SLEEFPirates.log($∂invlogitout[i])))
    else
        push!(loop_body.args, :($invlogitout = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout = $ninvlogitout * $invlogitout))
        push!(loop_body.args, :($mv[i] = SIMDPirates.vmuladd($(T(2)), $invlogitout, one($T))))
        push!(loop_body.args, :($log_jac += SLEEFPirates.log($∂invlogitout)))
    end

    push!(q.args, quote
        DistributionParameters.LoopVectorization.@vectorize $T for i ∈ 1:$L
            $loop_body
        end
        $out = LKJ_Correlation_Cholesky{$M}($mv)
        $θ += $N
        target += DistributionParameters.SIMDPirates.vsum($log_jac)
    end)
    push!(first_pass, q)
    if partial
        push!(second_pass, quote
            DistributionParameters.LoopVectorization.@vectorize $T for i ∈ 1:$N
                $∂θ[i] = one($T) - 2($invlogitout)[i] + ($(Symbol("###seed###", out)))[i] * ($∂invlogitout)[i]
            end
            $∂θ += $N
        end)
    end
    nothing
end
