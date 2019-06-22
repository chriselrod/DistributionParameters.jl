
abstract type AbstractLKJCorrCholesky{M,T,L} <: StructuredMatrices.AbstractLowerTriangularMatrix{M,T,L} end

struct LKJCorrCholesky{M,T,L} <: AbstractLKJCorrCholesky{M,T,L}
    data::NTuple{L,T}
end
struct PtrLKJCorrCholesky{M,T,L} <: AbstractLKJCorrCholesky{M,T,L}
    ptr::Ptr{T}
end
@generated PaddedMatrices.param_type_length(::Type{<: AbstractLKJCorrCholesky{M}}) where {M} = StructuredMatrices.binomial2(M)
@generated PaddedMatrices.param_type_length(::AbstractLKJCorrCholesky{M}) where {M} = StructuredMatrices.binomial2(M)


# struct Inverse_LKJ_Correlation_Cholesky{M,T,L} <: AbstractMatrix{T}
#     data::NTuple{T,L}
#     # inverse::NTuple{T,L}
# end
@inline Base.pointer(L::PtrLKJCorrCholesky) = L.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, L::PtrLKJCorrCholesky{M,T}) where {M,T} = L.ptr

abstract type AbstractLKJCholeskyConstraintAdjoint{P,T,L} <: AbstractArray{T,4} end
struct LKJCholeskyConstraintAdjoint{P,T,L} <: AbstractLKJCholeskyConstraintAdjoint{P,T,L}
    data::NTuple{L,T}
end
struct PtrLKJCholeskyConstraintAdjoint{P,T,L} <: AbstractLKJCholeskyConstraintAdjoint{P,T,L}
    ptr::Ptr{T}
end
@inline Base.pointer(A::PtrLKJCholeskyConstraintAdjoint) = A.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::PtrLKJCholeskyConstraintAdjoint{P,T}) where {P,T} = A.ptr
Base.size(::AbstractLKJCholeskyConstraintAdjoint{P}) where {P} = (P,P,P-1,P-1)

@inline function Base.getindex(adj::LKJCholeskyConstraintAdjoint{P,T,L}, i::Integer) where {P,T,L}
    @boundscheck i > L && PaddedMatrices.ThrowBoundsError("i = $i > $L.")
    @inbounds adj.data[i]
end
@inline function Base.getindex(A::Union{PtrLKJCorrCholesky{P,T,L},PtrLKJCholeskyConstraintAdjoint{P,T,L}}, i::Integer) where {P,T,L}
    @boundscheck i > L && PaddedMatrices.ThrowBoundsError("i = $i > $L.")
    VectorizationBase.load(A.ptr + (i-1)*sizeof(T))
end
@inline function Base.setindex!(A::Union{PtrLKJCorrCholesky{P,T,L},PtrLKJCholeskyConstraintAdjoint{P,T,L}}, v::T, i::Integer) where {P,T,L}
    @boundscheck i > L && PaddedMatrices.ThrowBoundsError("i = $i > $L.")
    VectorizationBase.store!(A.ptr + (i-1)*sizeof(T), v)
end
@inline function Base.getindex(adj::AbstractLKJCholeskyConstraintAdjoint{P,T,L}, i::CartesianIndex{4}) where {P,T,L}
    adj[i[1],i[2],i[3],i[4]]
end
function Base.getindex(adj::AbstractLKJCholeskyConstraintAdjoint{Mp1,T,L}, i, j, k, l) where {Mp1,T,L}
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

function LKJ_adjoint_mul_quote(Mp1,T)
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

@generated function Base.:*(
        t::LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractFixedSizePaddedVector{BP,T,BPL,BPL}},
        adj::AbstractLKJCholeskyConstraintAdjoint{Mp1,T,L}
    ) where {Mp1,T,L,BP,BPL}
    LKJ_adjoint_mul_quote(Mp1,T)
end

@generated function Base.:*(
        t::StructuredMatrices.AbstractLowerTriangularMatrix{Mp1,T},
        adj::AbstractLKJCholeskyConstraintAdjoint{Mp1,T,L}
    ) where {Mp1,T,L}
    LKJ_adjoint_mul_quote(Mp1,T)
end





"""
Generates the quote for the constraining transformation from z ∈ (-1,1) to LKJ_Correlation_Cholesky
without taking the derivative of the expression.
"""
function constrain_lkj_factor_quote(L, T, zsym, sp = false)
    # @show L
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
    logdetsym = gensym(:logdet)
    if M > 1
        push!(q.args, :($logdetsym = log( $(Expr(:call, :*, [Symbol(:ljp_,m) for m ∈ 3:M+1]...)) ) ))
    else
        push!(q.args, :($logdetsym = zero(T)))
    end
    lkj_length = StructuredMatrices.binomial2(Mp1+1)
    if sp
        lkjsym = gensym(:LKJ)
        push!(q.args, :($lkjsym = DistributionParameters.PtrLKJCorrCholesky{$Mp1,$T,$lkj_length}(pointer(sp,$T))))
        push!(q.args, :(sp += $(sizeof(T)*lkj_length)))
        i = 0
        for mc ∈ 1:M+1
            i += 1
            push!(q.args, :($lkjsym[$i] = $(Symbol(:x_, mc, :_, mc))))
        end
        for mc ∈ 1:M+1
            for mr ∈ mc+1:M+1
                i += 1
                push!(q.args, :($lkjsym[$i] = $(Symbol(:x_, mr, :_, mc))))
            end
        end
        return quote
            @fastmath @inbounds begin
                $q
            end, $lkjsym, logdetsym
        end              
    else
        output = Expr(:tuple,)
        for mc ∈ 1:M+1
            push!(output.args, Symbol(:x_, mc, :_, mc))
        end
        for mc ∈ 1:M+1
            for mr ∈ mc+1:M+1
                push!(output.args, Symbol(:x_, mr, :_, mc))
            end
        end
        return quote
            @fastmath @inbounds begin
                $q
            end
        end, :(DistributionParameters.LKJCorrCholesky{$Mp1,$T,$lkj_length}($output)), logdetsym
    end
end

∂lkj_sym(i, j, k) = Symbol(:∂x_, i, :_, j, :_∂z_, k)

"""
This function also generates code to calculate the gradient with respect to the log determinant of the Jacobian,
as well as the full Jacobian of the constraining transformation.
"""
function constrain_lkj_factor_jac_quote(L, T, zsym, sp = false)
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
k            end


        end
    end

    logdetsym = gensym(:logdet)
    if M > 1
        push!(q.args, :($logdetsym = log( $(Expr(:call, :*, [Symbol(:ljp_,m) for m ∈ 3:Mp1]...)) ) ))
    else
        push!(q.args, :($logdetsym = zero(T)))
    end

    if sp
        lkj_length = binomial2(Mp1+1)
        lkjsym = gensym(:LKJ)
        push!(q.args, :($lkjsym = PtrLKJCorrCholesky{$Mp1,$T,$lkj_length}(pointer(sp,$T))))
        push!(q.args, :(sp += $(sizeof(T)*lkj_length)))
        i = 0
        for mc ∈ 1:Mp1
            i += 1
            push!(q.args, :($lkjsym[$i] = $(Symbol(:x_, mc, :_, mc))))
        end
        for mc ∈ 1:Mp1
            for mr ∈ mc+1:Mp1
                i += 1
                push!(q.args, :($lkjsym[$i] = $(Symbol(:x_, mr, :_, mc))))
            end
        end
        Wm1 = VectorizationBase.pick_vector_width(Mp1, T) - 1
        LKJ_L1 = StructuredMatrices.binomial2(Mp1+1)
#        LKJ_L = (LKJ_L1 + Wm1) & ~Wm1
#        for i ∈ LKJ_L1+1:LKJ_L
#            push!(output.args, zero(T))
#        end


        ∂logdetsym = gensym(:∂logdet)
        bin2M = binomial2(M)
        push!(q.args, :($∂logdetsym = PtrVector{$bin2M,$T}(pointer(sp, $T))))
        push!(q.args, :(sp += $(sizeof(T)*bin2M)))
        i = 0
        for mc ∈ 1:M
            i += 1
            push!(q.args, :($∂logdetsym[$i] = zero($T)))
            for mr ∈ mc+2:M+1
                i += 1
                push!(q.args, :($∂logdetsym[$i] = $(Symbol(:∂ljp_, mr, :_, mc))))
            end
        end
#        ∂logdet_len = length(∂logdet.args)
 #       ∂logdet_len_full = (∂logdet_len + Wm1) & ~Wm1
 #       for i ∈ ∂logdet_len+1:∂logdet_len_full
 #           push!(∂logdet.args, zero(T))
 #       end

        jacobiansym = gensym(:jacobian)
        Ladj = 0
        for mp ∈ 1:M, mc ∈ mp+1:Mp1
            Ladj += 1
        end
        for mc ∈ 2:M, mp ∈ 1:mc, mr ∈ mc+1:Mp1
            Ladj += 1
        end
        push!(q.args, :($jacobiansym = PtrLKJCorrCholeskyAdjoint{$Mp1,$T,$Ladj}(pointer(sp,$T))))
        push!(q.args, :(sp += $(Ladj*sizeof(2))))
        # diagonal block of lkj
        i = 0
        for mp ∈ 1:M
            for mc ∈ mp+1:Mp1
                i +=1
                push!(q.args, :($jacobiansym[$i] = $(Symbol(:∂x_, mc, :_, mc, :_∂_, mp))))
            end
        end
        for mc ∈ 2:M
            for mp ∈ 1:mc
                for mr ∈ mc+1:Mp1 # iter through columns of z
                    i += 1
                    push!(q.args, :($jacobiansym[$i] = $(Symbol(:∂x_, mr, :_, mc, :_∂_, mp))))
                end
            end
        end
    # Return quote, followed by 4 outputs of the function
    # four outputs are:
    # 1. constrained
    # 2. logdeterminant of constraining transformation
    # 3. gradient with respect to log determinant
    # 4. Jacobian of the constraining transformation
    # constrain_q, constrained_expr, logdetsym, logdetgrad, jacobian
        return quote
            @fastmath @inbounds begin
                $q
            end
        end, lkjsym, logdetsym, ∂logdetsym, jacobiansym
    else    
        output = Expr(:tuple,)
        for mc ∈ 1:Mp1
            push!(output.args, Symbol(:x_, mc, :_, mc))
        end
        for mc ∈ 1:Mp1
            for mr ∈ mc+1:Mp1
                push!(output.args, Symbol(:x_, mr, :_, mc))
            end
        end
        Wm1 = VectorizationBase.pick_vector_width(Mp1, T) - 1
        LKJ_L1 = StructuredMatrices.binomial2(Mp1+1)
        LKJ_L = (LKJ_L1 + Wm1) & ~Wm1
        for i ∈ LKJ_L1+1:LKJ_L
            push!(output.args, zero(T))
        end


        ∂logdet = Expr(:tuple,)
        for mc ∈ 1:M
            push!(∂logdet.args, zero(T))
            for mr ∈ mc+2:M+1
                push!(∂logdet.args, Symbol(:∂ljp_, mr, :_, mc))
            end
        end
        ∂logdet_len = length(∂logdet.args)
        ∂logdet_len_full = (∂logdet_len + Wm1) & ~Wm1
        for i ∈ ∂logdet_len+1:∂logdet_len_full
            push!(∂logdet.args, zero(T))
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
        return quote
            @fastmath @inbounds begin
                $q
            end
        end, :(DistributionParameters.LKJ_Correlation_Cholesky{$Mp1,$T,$(LKJ_L)}($output)), logdetsym, :(DistributionParameters.ConstantFixedSizePaddedVector{$L}($∂logdet)), :(DistributionParameters.LKJCholeskyConstraintAdjoint{$Mp1,$T,$Ladj}($jacobian_tuple))
    end
end

@generated function lkj_constrain(zlkj::PaddedMatrices.AbstractFixedSizePaddedVector{L,T}) where {T,L}
    # L = StructuredMatrices.binomial2(M)
    lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym = constrain_lkj_factor_quote(L, T, :zlkj)
    quote
        $lkjconstrain_q
        $lkjconstrained_expr, $lkjlogdetsym
    end
end
@generated function ∂lkj_constrain(zlkj::PaddedMatrices.AbstractFixedSizePaddedVector{L,T}) where {L,T}
    # L = StructuredMatrices.binomial2(M)
    lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym, lkjlogdetgrad, lkjjacobian = constrain_lkj_factor_jac_quote(L, T, :zlkj)
    quote
        $lkjconstrain_q
        $lkjconstrained_expr, $lkjlogdetsym, $lkjlogdetgrad, $lkjjacobian
    end
end


@generated function lkj_constrain(sp::StackPointer, zlkj::PaddedMatrices.AbstractFixedSizePaddedVector{L,T}) where {T,L}
    # L = StructuredMatrices.binomial2(M)
    lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym = constrain_lkj_factor_quote(L, T, :zlkj, true)
    quote
        $lkjconstrain_q
        sp, ($lkjconstrained_expr, $lkjlogdetsym)
    end
end
@generated function ∂lkj_constrain(sp::StackPointer, zlkj::PaddedMatrices.AbstractFixedSizePaddedVector{L,T}) where {L,T}
    # L = StructuredMatrices.binomial2(M)
    lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym, lkjlogdetgrad, lkjjacobian = constrain_lkj_factor_jac_quote(L, T, :zlkj, true)
    quote
        $lkjconstrain_q
        sp, ($lkjconstrained_expr, $lkjlogdetsym, $lkjlogdetgrad, $lkjjacobian)
    end
end

@support_stack_pointer lkj_constrain
@support_stack_pointer ∂lkj_constrain


function load_parameter(first_pass, second_pass, out, ::Type{<: AbstractLKJCorrCholesky{M,T}}, partial = false) where {M,T}
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
    # @show N, L
    log_jac = gensym(:log_jac)
    zsym = gensym(:z) # z ∈ (-1, 1)
    q = quote
        $zsym = MutableFixedSizePaddedVector{$N,$T}(undef)
        # $log_jac = DistributionParameters.SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T))
        $log_jac = zero($T)
    end
    if partial
        push!(q.args, :($invlogitout = MutableFixedSizePaddedVector{$L,$T}(undef)))
        push!(q.args, :($∂invlogitout = MutableFixedSizePaddedVector{$L,$T}(undef)))
    end
    i = gensym(:i)
    loop_body = quote
        # $ninvlogitout = one($T) / (one($T) + SLEEFPirates.exp($(T(0.5)) * $θ[$i]))
        $ninvlogitout = one($T) / (one($T) + SLEEFPirates.exp($θ[$i]))
    end
    if partial
        push!(loop_body.args, :($invlogitout[$i] = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout[$i] = $ninvlogitout * $invlogitout[$i]))
        push!(loop_body.args, :($zsym[$i] = SIMDPirates.vmuladd($(T(-2)), $ninvlogitout, one($T))))
        push!(loop_body.args, :($log_jac += SLEEFPirates.log($∂invlogitout[$i])))
    else
        push!(loop_body.args, :($invlogitout = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout = $ninvlogitout * $invlogitout))
        push!(loop_body.args, :($zsym[$i] = SIMDPirates.vmuladd($(T(-2)), $ninvlogitout, one($T))))
        push!(loop_body.args, :($log_jac += SLEEFPirates.log($∂invlogitout)))
    end

    push!(q.args, quote
        ProbabilityModels.DistributionParameters.LoopVectorization.@vectorize $T for $i ∈ 1:$L
            $loop_body
        end
    end)
    lkjlogdetsym = gensym(:lkjlogdetsym)
    if partial
        # lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym, lkjlogdetgrad, lkjjacobian = constrain_lkj_factor_jac_quote(N, T, zsym)
        seedlkj = gensym(:seedlkj)
        lkjlogdetgradsym = gensym(:lkjlogdetgrad)
        lkjjacobiansym = gensym(:lkjjacobian)
        push!(second_pass, quote
            # $lkjlogdetgradsym = $lkjlogdetgrad
            # $lkjjacobiansym = $lkjjacobian
            # @show $zsym
            # @show $(Symbol("###seed###", out)) * $lkjjacobiansym
            # @show $lkjlogdetgradsym

            $seedlkj = ($(Symbol("###seed###", out)) * $lkjjacobiansym).parent # .parent to take off the transpose
            # @show $seedlkj'
            # @show $invlogitout'
            # @show $∂invlogitout'
            # ProbabilityModels.DistributionParameters.LoopVectorization.@vectorize $T for $i ∈ 1:$N
            #     $∂θ[$i] = $(T(0.5)) - ($invlogitout)[$i] + ($seedlkj)[$i] * ($∂invlogitout)[$i]
            #     # $∂θ[$i] = one($T) - $(T(2)) * ( ($invlogitout)[$i] - ($seedlkj)[$i] * ($∂invlogitout)[$i] )
            # end
            # println("invlogitout")
            # println($invlogitout)
            # println("seedlkj")
            # println($seedlkj)
            # println("lkjlogdetgradsym")
            # println($lkjlogdetgradsym)
            # println("∂invlogitout")
            # println($∂invlogitout)
            @vectorize for $i ∈ 1:$L
                # $∂θ[$i] = $(T(0.5)) - ($invlogitout)[$i] + (($seedlkj)[$i] + $lkjlogdetgradsym[$i]) * ($∂invlogitout)[$i]
                $∂θ[$i] = $(one(T)) - $(T(2)) * ( ($invlogitout)[$i] - (($seedlkj)[$i] + $lkjlogdetgradsym[$i]) * ($∂invlogitout)[$i] )
                # $∂θ[$i] = one($T) - $(T(2)) * ( ($invlogitout)[$i] - ($seedlkj)[$i] * ($∂invlogitout)[$i] )
            end
            $∂θ += $N
        end)
        push!(q.args, quote
            # $zsym = ConstantFixedSizePaddedVector{$M}($mv)
            # $lkjconstrain_q
            # $out = $lkjconstrained_expr
            $out, $lkjlogdetsym, $lkjlogdetgradsym, $lkjjacobiansym = DistributionParameters.∂lkj_constrain($zsym)
            $θ += $N
            # target += DistributionParameters.SIMDPirates.vsum($log_jac) + $lkjlogdetsym
            # target += $(T(0.5)) * ($log_jac + $lkjlogdetsym)
            # println("log_jac invlogit")
            # println($log_jac)
            # println("log_jac lkj")
            # println($lkjlogdetsym)
            target += ($log_jac +  $lkjlogdetsym)
        end)
        # @show second_pass
    else
        # lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym = constrain_lkj_factor_quote(N, T, zsym)

        push!(q.args, quote
            # $zsym = ConstantFixedSizePaddedVector{$M}($mv)
            # $lkjconstrain_q
            # $out = $lkjconstrained_expr
            $out, $lkjlogdetsym = DistributionParameters.lkj_constrain($zsym)
            $θ += $N
            # target += DistributionParameters.SIMDPirates.vsum($log_jac) + $lkjlogdetsym
            target += ($log_jac + $lkjlogdetsym)
        end)
    end
    # push!(q.args, :(@show $zsym))

    push!(first_pass, q)

    nothing
end
function load_parameter(sp::StackPointer, first_pass, second_pass, out, ::Type{<: AbstractLKJCorrCholesky{M,T}}, partial = false) where {M,T}
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
    # @show N, L
    log_jac = gensym(:log_jac)
    zsym = gensym(:z) # z ∈ (-1, 1)
    q = quote
        (sp, $zsym) = PtrVector{$N,$T}(sp)
#        $zsym = MutableFixedSizePaddedVector{$N,$T}(undef)
        # $log_jac = DistributionParameters.SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T))
        $log_jac = zero($T)
    end
    if partial
        push!(q.args, :($invlogitout = MutableFixedSizePaddedVector{$L,$T}(undef)))
        push!(q.args, :($∂invlogitout = MutableFixedSizePaddedVector{$L,$T}(undef)))
    end
    i = gensym(:i)
    loop_body = quote
        # $ninvlogitout = one($T) / (one($T) + SLEEFPirates.exp($(T(0.5)) * $θ[$i]))
        $ninvlogitout = one($T) / (one($T) + SLEEFPirates.exp($θ[$i]))
    end
    if partial
        push!(loop_body.args, :($invlogitout[$i] = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout[$i] = $ninvlogitout * $invlogitout[$i]))
        push!(loop_body.args, :($zsym[$i] = SIMDPirates.vmuladd($(T(-2)), $ninvlogitout, one($T))))
        push!(loop_body.args, :($log_jac += SLEEFPirates.log($∂invlogitout[$i])))
    else
        push!(loop_body.args, :($invlogitout = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout = $ninvlogitout * $invlogitout))
        push!(loop_body.args, :($zsym[$i] = SIMDPirates.vmuladd($(T(-2)), $ninvlogitout, one($T))))
        push!(loop_body.args, :($log_jac += SLEEFPirates.log($∂invlogitout)))
    end

    push!(q.args, quote
        ProbabilityModels.DistributionParameters.LoopVectorization.@vectorize $T for $i ∈ 1:$L
            $loop_body
        end
    end)
    lkjlogdetsym = gensym(:lkjlogdetsym)
    if partial
        # lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym, lkjlogdetgrad, lkjjacobian = constrain_lkj_factor_jac_quote(N, T, zsym)
        seedlkj = gensym(:seedlkj)
        lkjlogdetgradsym = gensym(:lkjlogdetgrad)
        lkjjacobiansym = gensym(:lkjjacobian)
        push!(second_pass, quote
            # $lkjlogdetgradsym = $lkjlogdetgrad
            # $lkjjacobiansym = $lkjjacobian
            # @show $zsym
            # @show $(Symbol("###seed###", out)) * $lkjjacobiansym
            # @show $lkjlogdetgradsym

            $seedlkj = ($(Symbol("###seed###", out)) * $lkjjacobiansym).parent # .parent to take off the transpose
            # @show $seedlkj'
            # @show $invlogitout'
            # @show $∂invlogitout'
            # ProbabilityModels.DistributionParameters.LoopVectorization.@vectorize $T for $i ∈ 1:$N
            #     $∂θ[$i] = $(T(0.5)) - ($invlogitout)[$i] + ($seedlkj)[$i] * ($∂invlogitout)[$i]
            #     # $∂θ[$i] = one($T) - $(T(2)) * ( ($invlogitout)[$i] - ($seedlkj)[$i] * ($∂invlogitout)[$i] )
            # end
            # println("invlogitout")
            # println($invlogitout)
            # println("seedlkj")
            # println($seedlkj)
            # println("lkjlogdetgradsym")
            # println($lkjlogdetgradsym)
            # println("∂invlogitout")
            # println($∂invlogitout)
            @vectorize for $i ∈ 1:$L
                # $∂θ[$i] = $(T(0.5)) - ($invlogitout)[$i] + (($seedlkj)[$i] + $lkjlogdetgradsym[$i]) * ($∂invlogitout)[$i]
                $∂θ[$i] = $(one(T)) - $(T(2)) * ( ($invlogitout)[$i] - (($seedlkj)[$i] + $lkjlogdetgradsym[$i]) * ($∂invlogitout)[$i] )
                # $∂θ[$i] = one($T) - $(T(2)) * ( ($invlogitout)[$i] - ($seedlkj)[$i] * ($∂invlogitout)[$i] )
            end
            $∂θ += $N
        end)
        push!(q.args, quote
            # $zsym = ConstantFixedSizePaddedVector{$M}($mv)
            # $lkjconstrain_q
            # $out = $lkjconstrained_expr
            $out, $lkjlogdetsym, $lkjlogdetgradsym, $lkjjacobiansym = DistributionParameters.∂lkj_constrain($zsym)
            $θ += $N
            # target += DistributionParameters.SIMDPirates.vsum($log_jac) + $lkjlogdetsym
            # target += $(T(0.5)) * ($log_jac + $lkjlogdetsym)
            # println("log_jac invlogit")
            # println($log_jac)
            # println("log_jac lkj")
            # println($lkjlogdetsym)
            target += ($log_jac +  $lkjlogdetsym)
        end)
        # @show second_pass
    else
        # lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym = constrain_lkj_factor_quote(N, T, zsym)

        push!(q.args, quote
            # $zsym = ConstantFixedSizePaddedVector{$M}($mv)
            # $lkjconstrain_q
            # $out = $lkjconstrained_expr
            $out, $lkjlogdetsym = DistributionParameters.lkj_constrain($zsym)
            $θ += $N
            # target += DistributionParameters.SIMDPirates.vsum($log_jac) + $lkjlogdetsym
            target += ($log_jac + $lkjlogdetsym)
        end)
    end
    # push!(q.args, :(@show $zsym))

    push!(first_pass, q)

    nothing
end

function load_parameter(first_pass, second_pass, out, ::Type{<: AbstractLKJCorrCholesky{M}}, partial = false) where {M}
    load_parameter(first_pass, second_pass, out, LKJ_Correlation_Cholesky{M,Float64}, partial)
end
