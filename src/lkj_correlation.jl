
struct LKJ_Correlation_Cholesky{M,T,L} <: AbstractMatrix{T}
    data::NTuple{T,L}
    # inverse::NTuple{T,L}
end
# struct Inverse_LKJ_Correlation_Cholesky{M,T,L} <: AbstractMatrix{T}
#     data::NTuple{T,L}
#     # inverse::NTuple{T,L}
# end


struct LKJCholeskyConstraintAdjoint{P,T,L} <: AbstractArray{T,4}
    data::NTuple{L,T}
end
Base.size(::LKJCholeskyConstraintAdjoint{P}) where {P} = (P,P,P-1,P-1)

@inline function Base.getindex(adj::LKJCholeskyConstraintAdjoint{P,T,L}, i) where {P,T,L}
    @boundscheck i > L && PaddedMatrices.ThrowBoundsError("i = $i > $L.")
    @inbounds adj.data[i]
end
function Base.getindex(adj::LKJCholeskyConstraintAdjoint{Mp1,T,L}, i, j, k, l) where {Mp1,T,L}
    # Welcome to Branchapalooza!!!
    kp1 = k + 1
    M = Mp1 - 1
    @boundscheck max(i,j,kp1,l+1) > Mp1 && PaddedMatrices.ThrowBoundsError("At least one of (i,j,k,l) = ($i,$j,$k,$l) > ($Mp1,$Mp1,$M,$M)")
    # Lets try and skip trudging through the loop
    j > i && return zero(T)
    l > k && return zero(T)
    i == kp1 || return zero(T)
    i == j == 1 && return zero(T)
    # note order matters.
    # if i == 1, return zero(T), or if
    # if i != k + 1, return zero(T)
    l == j == 1 && return one(T)# + 10l + 100j
    if i == j == kp1
        ind = 0
        # This for loop can be removed
        # I'll get around to it
        # but this function is not performance critical.
        for mp ∈ 1:M
            if mp == l
                return @inbounds adj[ind + i - mp]
            else
                ind += Mp1 - mp
            end
        end
    end
    ind = StructuredMatrices.binomial2(Mp1)
    for mc ∈ 2:M # columns of output (LKJ)
        mc_equal_j = mc == j
        for mp ∈ 1:mc #
            mp_equal_l = mp == l
            for mr ∈ mc+1:Mp1 # iter through columns of z
                ind += 1
                if mc_equal_j && mp_equal_l && (mr == i)
                    return @inbounds adj[ind]
                end
            end
        end
    end
    zero(T)
end

@generated function Base.:*(
        t::LinearAlgebra.Adjoint{T,PaddedMatrices.AbstractFixedSizePaddedVector{BP,T,BPL,BPL}},
        adj::LKJCholeskyConstraintAdjoint{Mp1,T,L}
    ) where {Mp1,T,L,BP,BPL}

    M = Mp1 - 1
    q = quote end
    StructuredMatrices.load_packed_L_quote!(q.args, Mp1, :∂target_∂L, :t)

    # first, we initialize all outputs with the diagonal pass
    # for the partials with respect to column 1 of the inputs,
    # we also add the identity matrix part of the jacobian
    # (ie, partials with respect to the first column are "1")
    for mc ∈ 1:M
        push!(q.args, :(
            $(Symbol(:∂lkj_∂z_, mc, :_, 1)) = $(PaddedMatrices.sym(:∂target_∂L, mc+1, 1)) +
                                        $(PaddedMatrices.sym(:∂target_∂L, mc+1, mc+1)) * adj[$mc]
                        )
        )
    end
    # partials with respec to remaining columns of the inputs
    ind = M
    for mp ∈ 2:M
        for mc ∈ mp:M
            ind += 1
            push!(q.args, :($(Symbol(:∂lkj_∂z_, mc, :_, mp)) = $(PaddedMatrices.sym(:∂target_∂L, mc+1, mc+1)) * adj[$ind] ))
        end
    end
    # now that all partials are initialized, we update them.
    for mc ∈ 2:M
        for mp ∈ 1:mc
            for mr ∈ mc+1:Mp1 # iter through columns of z
                ind += 1
                push!(q.args, :($(Symbol(:∂lkj_∂z_, mr-1, :_, mp)) += $(PaddedMatrices.sym(:∂target_∂L, mr, mc)) * adj[$ind] ))
            end
        end
    end


    outtup = Expr(:tuple,)
    # for p ∈ 1:M
    #     push!(outtup.args, PaddedMatrices.sym(:∂lkj_∂z, p, p) )
    # end
    for pc ∈ 1:M
        for pr ∈ pc:M
            push!(outtup.args, PaddedMatrices.sym(:∂lkj_∂z, pr, pc))
        end
    end
    lkj_l = StructuredMatrices.binomial2(Mp1)
    lkj_l_full = PaddedMatrices.pick_L(lkj_l, T)
    for p ∈ StructuredMatrices.binomial2(Mp1)+1:lkj_l_full
        push!(outtup.args, zero(T))
    end

    quote
        @fastmath @inbounds begin
            $q
            ConstantFixedSizePaddedArray{Tuple{$lkj_l}, $T, 1, $lkj_l_full, $lkj_l_full}($outtup)'
        end
    end
end





"""
Generates the quote for the constraining transformation from z ∈ (-1,1) to LKJ_Correlation_Cholesky
without taking the derivative of the expression.
"""
function constrain_lkj_factor_quote(L, T, zsym)
    M = (Int(sqrt(1 + 8L))-1)>>1
    Mp1 = M+1
    q = quote x_1_1 = one($T) end
    for m ∈ 1:M
        xm = Symbol(:x_, m+1, :_, 1)
        push!(q.args, :( $xm = $zsym[$m]) )
        push!(q.args, :($(Symbol(:Omx²_, m+1, :_, 2)) = 1 - $xm * $xm))
    end
    ljnum = 0
    zind = M
    for mc ∈ 2:Mp1
        push!(q.args, :($(Symbol(:x_, mc, :_, mc)) = sqrt($(Symbol(:Omx²_, mc, :_, mc)))))
        for mr ∈ mc+1:Mp1
            xm = Symbol(:x_, mr, :_, mc)
            Omx²m = Symbol(:Omx²_, mr, :_, mc)
            ljnum += 1
            zind += 1
            lj = Symbol(:lj_, mr, :_, mc)
            # We're taking the derivative with respect to 0.5log(Omx²m) for each of the Omx²m encountered in this loop.
            push!(q.args, :($lj = sqrt($Omx²m) ))
            push!(q.args, :($xm = $zsym[$zind] * $lj ))
            push!(q.args, :($(Symbol(:Omx²_, mr, :_, mc+1)) = $Omx²m - $xm * $xm ))
            if mc == 2
                push!(q.args, :( $(Symbol(:ljp_, mr)) = $lj))
            else
                push!(q.args, :( $(Symbol(:ljp_, mr)) *= $lj))
            end
        end
    end
    output = Expr(:tuple,)
    for mc ∈ 1:M+1
        push!(output.args, Symbol(:x_, mc, :_, mc))
    end
    for mc ∈ 1:M+1
        for mr ∈ mc+1:M+1
            push!(output.args, Symbol(:x_, mr, :_, mc))
        end
    end
    logdetsym = gensym(:logdet)
    if M > 1
        push!(q.args, :($logdetsym = log( $(Expr(:call, :*, [Symbol(:ljp_,m) for m ∈ 3:M+1]...)) ) ))
    else
        push!(q.args, :($logdetsym = zero(T)))
    end
    quote
        @fastmath @inbounds begin
            $q
        end
    end, :(LKJ_Correlation_Cholesky{$Mp1,$T,$(binomial2(Mp1+1))}($output)), logdetsym
end

∂lkj_sym(i, j, k) = Symbol(:∂x_, i, :_, j, :_∂z_, k)

"""
This function also generates code to calculate the gradient with respect to the log determinant of the Jacobian,
as well as the full Jacobian of the constraining transformation.
"""
function constrain_lkj_factor_jac_quote(L, T, zsym)
    # partial = true
    # Storage is the subdiagonal lower triangle
    # (the diagonal itself is determined by the correlation matrix constraint)
    #
    # The correlation matrix is (M+1) x (M+1)
    # our subdiagonal triangle is M x M
    #
    # We denote the (-1, 1) constrained parameters "z"
    # while "x" is the (M+1) x (M+1) cholesky factor of the correlation matrix.
    M = (Int(sqrt(1 + 8L))-1)>>1
    Mp1 = M+1
    q = quote x_1_1 = one($T) end
    for m ∈ 1:M
        xm = Symbol(:x_, m+1, :_, 1)
        Omx²m = Symbol(:Omx²_, m+1, :_2)
        push!(q.args, :( $xm = $zsym[$m]) )
        push!(q.args, :( $Omx²m = one($T) - $xm * $xm ) )
        push!(q.args, :($(Symbol(:∂x_, m+1, :_1_∂_1)) = one($T) ) )
    end
    ljnum = 0
    zind = M
    for mc ∈ 2:Mp1
        xm = Symbol(:x_, mc, :_, mc)

        # Note that the partials terminate at 1 less, because "z" only fills the
        # lower subdiagonal triangle. Iteration stops 2 early, as these are ones that reference
        # previous calculations. Final is the first partial for that z-term
        # (that z term would be the furthest to the right in its given row)
        xm_prev = Symbol(:x_, mc, :_, mc-1) # update based on that z term
        Omx²m = Symbol(:Omx²_, mc, :_, mc); Omx²_previous = Symbol(:Omx²_, mc, :_, mc-1)
        mc > 2 && push!(q.args, :($Omx²m = $Omx²_previous - $xm_prev * $xm_prev))
        push!(q.args, :($xm = sqrt($Omx²m)))
        for mc_nested ∈ 1:mc-2
            ∂Omx²_previous = Symbol(:∂, Omx²_previous, :_, mc_nested)
            push!(q.args, :($(Symbol(:∂, Omx²m, :_, mc_nested)) = $∂Omx²_previous -2*$xm_prev * $(Symbol(:∂, xm_prev, :_∂_, mc_nested))   ))
        end
        let mc_nested = mc - 1 # initialize
            push!(q.args, :($(Symbol(:∂, Omx²m, :_, mc_nested)) = -2*$xm_prev * $(Symbol(:∂, xm_prev, :_∂_, mc_nested)) ))
        end
        # Now determin ∂x/∂z
        for mc_nested ∈ 1:mc-1 # partial is a function of previous values
            ∂xm = Symbol(:∂, xm, :_∂_, mc_nested)
            ∂Omx²m = Symbol(:∂, Omx²m, :_, mc_nested)
            push!(q.args, :($∂xm = $(T(0.5)) * $xm * $∂Omx²m / $Omx²m ) )
        end

        for mr ∈ mc+1:Mp1
            xm = Symbol(:x_, mr, :_, mc)
            xm_prev = Symbol(:x_, mr, :_, mc-1)
            Omx²_previous = Symbol(:Omx²_, mr, :_, mc-1)
            Omx²m = Symbol(:Omx²_, mr, :_, mc)
            ljnum += 1
            zind += 1
            lj = Symbol(:lj_, mr, :_, mc)
            mc > 2 && push!(q.args, :($Omx²m = $Omx²_previous - $xm_prev * $xm_prev ))
            # We're taking the derivative with respect to 0.5log(Omx²m) for each of the Omx²m encountered in this loop.
            push!(q.args, :($lj = sqrt($Omx²m) ))
            push!(q.args, :($xm = $zsym[$zind] * $lj ))
            if mc == 2
                push!(q.args, :( $(Symbol(:ljp_, mr)) = $lj))
            else
                push!(q.args, :( $(Symbol(:ljp_, mr)) *= $lj))
            end

            # Now we calculate partials with respect to Omx²m from this collomn
            for mc_nested ∈ 1:mc-2
                ∂Omx²_previous = Symbol(:∂, Omx²_previous, :_, mc_nested)
                push!(q.args, :($(Symbol(:∂, Omx²m, :_, mc_nested)) = $∂Omx²_previous -2*$xm_prev * $(Symbol(:∂, xm_prev, :_∂_, mc_nested))   ))
                push!(q.args, :($(Symbol(:∂ljp_, mr, :_, mc_nested)) += $(T(0.5)) * $(Symbol(:∂, Omx²m, :_, mc_nested)) / $Omx²m ))
            end
            let mc_nested = mc - 1 # initialize
                push!(q.args, :($(Symbol(:∂, Omx²m, :_, mc_nested)) = -2*$xm_prev * $(Symbol(:∂, xm_prev, :_∂_, mc_nested)) ))
                push!(q.args, :($(Symbol(:∂ljp_, mr, :_, mc_nested)) = $(T(0.5)) * $(Symbol(:∂, Omx²m, :_, mc_nested)) / $Omx²m ))
            end

            for mc_nested ∈ 1:mc-1 # partial is a function of previous values
                ∂xm = Symbol(:∂, xm, :_∂_, mc_nested)
                ∂Omx²m = Symbol(:∂, Omx²m, :_, mc_nested)
                push!(q.args, :($∂xm = $(T(0.5)) * $xm * $∂Omx²m / $Omx²m ) )
            end
            let mc_nested = mc # new partial
                ∂xm = Symbol(:∂, xm, :_∂_, mc_nested)
                push!(q.args, :($∂xm = $lj ) )
            end


        end
    end
    output = Expr(:tuple,)
    for mc ∈ 1:M+1
        push!(output.args, Symbol(:x_, mc, :_, mc))
    end
    for mc ∈ 1:M+1
        for mr ∈ mc+1:M+1
            push!(output.args, Symbol(:x_, mr, :_, mc))
        end
    end
    ∂logdet = Expr(:tuple,)
    for mc ∈ 1:M
        push!(∂logdet.args, zero(T))
        for mr ∈ mc+2:M+1
            push!(∂logdet.args, Symbol(:∂ljp_, mr, :_, mc))
        end
    end
    logdetsym = gensym(:logdet)
    if M > 1
        push!(q.args, :($logdetsym = log( $(Expr(:call, :*, [Symbol(:ljp_,m) for m ∈ 3:Mp1]...)) ) ))
    else
        push!(q.args, :($logdetsym = zero(T)))
    end

    jacobian_tuple = Expr(:tuple,)
    # diagonal block of lkj
    for mp ∈ 1:M
        for mc ∈ mp+1:Mp1
            push!(jacobian_tuple.args, Symbol(:∂x_, mc, :_, mc, :_∂_, mp))
        end
    end
    for mc ∈ 2:M
        for mp ∈ 1:mc
            for mr ∈ mc+1:Mp1 # iter through columns of z
                push!(jacobian_tuple.args, Symbol(:∂x_, mr, :_, mc, :_∂_, mp))
            end
        end
    end
    Lbase = length(jacobian_tuple.args)
    Ladj = PaddedMatrices.pick_L(Lbase, T)
    for i ∈ Lbase+1:Ladj
        push!(jacobian_tuple.args, zero(T))
    end
    # Return quote, followed by 4 outputs of the function
    # four outputs are:
    # 1. constrained
    # 2. logdeterminant of constraining transformation
    # 3. gradient with respect to log determinant
    # 4. Jacobian of the constraining transformation
    # constrain_q, constrained_expr, logdetsym, logdetgrad, jacobian
    quote
        @fastmath @inbounds begin
            $q
        end
    end, :(LKJ_Correlation_Cholesky{$Mp1,$T,$(binomial2(Mp1+1))}($output)), logdetsym, :(ConstantFixedSizePaddedVector{$L}($∂logdet)), :(LKJCholeskyConstraintAdjoint{$Mp1,$T,$Ladj}($jacobian_tuple))
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
    zsym = gensym(:z) # z ∈ (-1, 1)
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
    end)
    if partial
        lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym, lkjlogdetgrad, lkjjacobian = constrain_lkj_factor_jac_quote(L, T, zsym)
        seedlkj = gensym(:seedlkj)
        push!(second_pass, quote
            $seedlkj = $lkjlogdetgrad + ($(Symbol("###seed###", out)) * $lkjjacobian).parent # .patent to take off the transpose
            DistributionParameters.LoopVectorization.@vectorize $T for i ∈ 1:$N
                $∂θ[i] = one($T) - 2($invlogitout)[i] + ($seedlkj)[i] * ($∂invlogitout)[i]
            end
            $∂θ += $N
        end)
    else
        lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym = constrain_lkj_factor_quote(L, T, zsym)
    end

    push!(q.args, quote
        $zsym = ConstantFixedSizePaddedVector{$M}($mv)
        $lkjconstrain_q
        $out = $lkjconstrained_expr
        $θ += $N
        target += DistributionParameters.SIMDPirates.vsum($log_jac) + $lkjlogdetsym
    end)
    push!(first_pass, q)

    nothing
end
