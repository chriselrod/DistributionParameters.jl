using StructuredMatrices: binomial2



### Data layout
"""
Intention is for default data layout of CorrCholesky to be
Diagonals [aligned]
Lower sub-diagonal triangle (packed column major)
log(Diagonals) [aligned]
inv(Diagonals) [aligned] -- OPTIONAL, default absent
"""
mutable struct CorrCholesky{M,T,L} <: StructuredMatrices.AbstractMutableLowerTriangularMatrix{M,T,L}
    data::NTuple{L,T}
    CorrCholesky{M,T,L}(::UndefInitializer) where {M,T,L} = new{M,T,L}()
end
struct PtrCorrCholesky{M,T,L} <: StructuredMatrices.AbstractMutableLowerTriangularMatrix{M,T,L}
    ptr::Ptr{T}
end
"""
Intention is for default data layout of CorrCholesky to be
Diagonals [aligned]
Lower sub-diagonal triangle (packed column major)
log(Diagonals) [aligned]
inv(Diagonals) [aligned]
"""
mutable struct CovarCholesky{M,T,L} <: StructuredMatrices.AbstractMutableLowerTriangularMatrix{M,T,L}
    data::NTuple{L,T}
    CovarCholesky{M,T,L}(::UndefInitializer) where {M,T,L} = new{M,T,L}()
end
struct PtrCovarCholesky{M,T,L} <: StructuredMatrices.AbstractMutableLowerTriangularMatrix{M,T,L}
    ptr::Ptr{T}
end

const AbstractCorrCholesky{M,T,L} = Union{CorrCholesky{M,T,L},PtrCorrCholesky{M,T,L}}
const AbstractCovarCholesky{M,T,L} = Union{CovarCholesky{M,T,L},PtrCovarCholesky{M,T,L}}

@generated function PtrCovarCholesky{M,T}(sp::StackPointer) where {M,T}
    L = VectorizationBase.align(VectorizationBase.align(VectorizationBase.align(binomial2(M+1), T) + M, T) + M, T)
    quote
        $(Expr(:meta,:inline))
        sp + $(L*sizeof(T)), PtrCovarCholesky{$M,$T,$L}(pointer(sp, T))
    end
end

@generated function ReverseDiffExpressionsBase.alloc_adjoint(C::Union{<:AbstractCorrCholesky{M,T},<:AbstractCovarCholesky{M,T}}) where {M,T}
    L = PaddedMatrices.calc_padding(StructuredMatrices.binomial2(M+1), T)
    quote
        $(Expr(:meta,:inline))
        StructuredMatrices.PtrLowerTriangularMatrix{$M,$T,$L}(SIMDPirates.alloca(Val{$L}(),$T))
    end
end
@inline ReverseDiffExpressionsBase.alloc_adjoint(sp::StackPointer, C::AbstractCorrCholesky{M,T}) where {M,T} = StructuredMatrices.PtrLowerTriangularMatrix{M,T}(sp)
@inline ReverseDiffExpressionsBase.alloc_adjoint(sp::StackPointer, C::AbstractCovarCholesky{M,T}) where {M,T} = StructuredMatrices.PtrLowerTriangularMatrix{M,T}(sp)

function caches_logdiag(M, L, ::Type{T} = Float64) where {T}
    offset_length = VectorizationBase.align(binomial2(M+1),T)
    L >= offset_length + M, offset_length
end
function caches_invdiag(M, L, ::Type{T} = Float64) where {T}
    offset_length = VectorizationBase.align(VectorizationBase.align(binomial2(M+1),T) + M, T)
    L >= offset_length + M, offset_length
end
@inline logdiag(C::StructuredMatrices.AbstractDiagTriangularMatrix{M,T,L}) where {M,T,L} = PaddedMatrices.LazyMap(SLEEFPirates.log, PtrVector{M,T,M,true}(pointer(C)))
@inline invdiag(C::StructuredMatrices.AbstractDiagTriangularMatrix{M,T,L}) where {M,T,L} = PaddedMatrices.LazyMap(SIMDPirates.vinv, PtrVector{M,T,M,true}(pointer(C)))
@generated function logdiag(C::Union{<:AbstractCorrCholesky{M,T,L},<:AbstractCovarCholesky{M,T,L}}) where {M,T,L}
    c, offset = caches_logdiag(M,L)
    if c
        quote
            $(Expr(:meta,:inline))
            PtrVector{$M,$T,$(PaddedMatrices.calc_padding(M,T)),false}(pointer(C) + $(offset*sizeof(T)))
        end
    else
        quote
            $(Expr(:meta,:inline))
            d = PtrVector{$M,$T,$M,true}(pointer(C))
            PaddedMatrices.LazyMap(SLEEFPirates.log, d)
        end
    end
end
@generated function invdiag(C::Union{<:AbstractCorrCholesky{M,T,L},<:AbstractCovarCholesky{M,T,L}}) where {M,T,L}
    c, offset = caches_invdiag(M, L)
    if c
        quote
            $(Expr(:meta,:inline))
            PtrVector{$M,$T,$(PaddedMatrices.calc_padding(M,T)),false}(pointer(C) + $(offset*sizeof(T)))
        end
    else
        quote
            $(Expr(:meta,:inline))
            d = PtrVector{$M,$T,$M,true}(pointer(C))
            PaddedMatrices.LazyMap(SIMDPirates.vinv, d)
        end
    end
end


@generated PaddedMatrices.type_length(::Type{<: AbstractCorrCholesky{M}}) where {M} = StructuredMatrices.binomial2(M+1)
@generated PaddedMatrices.type_length(::AbstractCorrCholesky{M}) where {M} = StructuredMatrices.binomial2(M+1)
@generated PaddedMatrices.param_type_length(::Type{<: AbstractCorrCholesky{M}}) where {M} = StructuredMatrices.binomial2(M)
@generated PaddedMatrices.param_type_length(::AbstractCorrCholesky{M}) where {M} = StructuredMatrices.binomial2(M)

@inline Base.pointer(L::Union{CorrCholesky{M,T},CovarCholesky{M,T}}) where {M,T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(L))
@inline Base.pointer(L::Union{PtrCorrCholesky,PtrCovarCholesky}) = L.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, L::PtrCorrCholesky{M,T}) where {M,T} = L.ptr

abstract type AbstractCholeskyConstraintAdjoint{P,T,L} <: AbstractArray{T,4} end
mutable struct CholeskyConstraintAdjoint{P,T,L} <: AbstractCholeskyConstraintAdjoint{P,T,L}
    data::NTuple{L,T}
    CholeskyConstraintAdjoint{P,T,L}(::UndefInitializer) where {P,T,L} = new()
end
struct PtrCholeskyConstraintAdjoint{P,T,L} <: AbstractCholeskyConstraintAdjoint{P,T,L}
    ptr::Ptr{T}
end
@inline VectorizationBase.vectorizable(A::Union{PtrCorrCholesky,PtrCovarCholesky}) = VectorizationBase.Pointer(A.ptr)
@inline Base.pointer(A::PtrCholeskyConstraintAdjoint) = A.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::PtrCholeskyConstraintAdjoint{P,T}) where {P,T} = A.ptr
Base.size(::AbstractCholeskyConstraintAdjoint{P}) where {P} = (P,P,P-1,P-1)

@inline function Base.getindex(adj::CholeskyConstraintAdjoint{P,T,L}, i::Integer) where {P,T,L}
    @boundscheck i > L && PaddedMatrices.ThrowBoundsError("i = $i > $L.")
    @inbounds adj.data[i]
end
@inline function Base.getindex(A::Union{PtrCorrCholesky{P,T,L},PtrCholeskyConstraintAdjoint{P,T,L}}, i::Integer) where {P,T,L}
    @boundscheck i > L && PaddedMatrices.ThrowBoundsError("i = $i > $L.")
    VectorizationBase.load(A.ptr + (i-1)*sizeof(T))
end
@inline function Base.setindex!(A::Union{PtrCorrCholesky{P,T,L},PtrCholeskyConstraintAdjoint{P,T,L}}, v::T, i::Integer) where {P,T,L}
    @boundscheck i > L && PaddedMatrices.ThrowBoundsError("i = $i > $L.")
    VectorizationBase.store!(A.ptr + (i-1)*sizeof(T), v)
end
@inline function Base.getindex(adj::AbstractCholeskyConstraintAdjoint{P,T,L}, i::CartesianIndex{4}) where {P,T,L}
    adj[i[1],i[2],i[3],i[4]]
end
function Base.getindex(adj::AbstractCholeskyConstraintAdjoint{Mp1,T,L}, i, j, k, l) where {Mp1,T,L}
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
    for mc ∈ 2:M # columns of output ()
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

function corr_cholesky_adjoint_mul_quote(Mp1, T, add::Bool)#,sp::Bool = false)
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
    lkj_l = StructuredMatrices.binomial2(Mp1)
    lkj_l_full = PaddedMatrices.calc_padding(lkj_l, T)
    push!(q.args, :(sptr = pointer(out)))
    ind = 0
    size_T = sizeof(T)
    for pc ∈ 1:M
        for pr ∈ pc:M
            sptrind = :(sptr + $size_T*$ind)
            ∂lkjsym = PaddedMatrices.sym(:∂lkj_∂z, pr, pc)
            if add
                push!(q.args, :(VectorizationBase.store!($sptrind, Base.FastMath.add_fast(VectorizationBase.load($sptrind), $∂lkjsym))))
            else
                push!(q.args, :(VectorizationBase.store!($sptrind, $∂lkjsym)))
            end
            ind += 1
        end
    end
    push!(q.args, :out)
    quote
        @fastmath @inbounds begin
            $q
        end
    end
end

# @generated function LinearAlgebra.mul!(
#     c::AbstractLowerTriangularMatrix{Mp1,T},
#     t::LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractFixedSizeVector{BP,T,BPL}},
#     adj::AbstractCholeskyConstraintAdjoint{Mp1,T,L}
# ) where {Mp1,T,L,BP,BPL}
#     corr_cholesky_adjoint_mul_quote(Mp1,T)
# end

@generated function LinearAlgebra.mul!(
    out::StructuredMatrices.AbstractLowerTriangularMatrix{M,T},
    t::StructuredMatrices.AbstractLowerTriangularMatrix{Mp1,T},
    adj::AbstractCholeskyConstraintAdjoint{Mp1,T,L}
) where {Mp1,T,L,M}
    corr_cholesky_adjoint_mul_quote(Mp1,T,false)
end
@generated function PaddedMatrices.muladd!(
    out::StructuredMatrices.AbstractLowerTriangularMatrix{M,T},
    t::StructuredMatrices.AbstractLowerTriangularMatrix{Mp1,T},
    adj::AbstractCholeskyConstraintAdjoint{Mp1,T,L}
) where {Mp1,T,L,M}
    corr_cholesky_adjoint_mul_quote(Mp1,T,true)
end



function add_calc_logdet_expr!(q, M, ::Type{T}, logdetsym) where {T}
    Mp1 = M + 1; Mm1 = M - 1
    if M < 2
        push!(q.args, :($logdetsym = zero(T)))
    elseif Mp1 < 10
        push!(q.args, :($logdetsym = log( $(Expr(:call, :(Base.FastMath.mul_fast), [Symbol(:ljp_,m) for m ∈ 3:Mp1]...)) ) ))
    else
        W, Wshift = VectorizationBase.pick_vector_width_shift(Mm1, T)
        ljp_reps = Mm1 >>> Wshift
        ljp_rem = Mm1 & (W - 1)
        mincr = 2
        push!(q.args, :($logdetsym = SLEEFPirates.log($(Expr(:tuple, [:(Core.VecElement{$T}($(Symbol(:ljp_,w+mincr)))) for w ∈ 1:W]...)))))
        for mr ∈ 1:ljp_reps
            mincr += W
            push!(q.args, :($logdetsym = vadd($logdetsym, SLEEFPirates.log($(Expr(:tuple, [:(Core.VecElement{$T}($(Symbol(:ljp_,w+mincr)))) for w ∈ 1:W]...))))))
        end
        if ljp_rem > 0
            mincr += W
            push!(q.args, :($logdetsym = vadd($logdetsym, SLEEFPirates.log($(Expr(:tuple, [:(Core.VecElement{$T}($(w > Mp1 ? one(T) : Symbol(:ljp_,w)))) for w ∈ 1+mincr:W+mincr]...))))))
        end
    end
    nothing
end


"""
Generates the quote for the constraining transformation from z ∈ (-1,1) to _Correlation_Cholesky
without taking the derivative of the expression.
"""
function constrain_lkj_factor_quote(
    Mp1::Int, L::Int, LL::Int, ::Type{T}, lkjsym::Symbol, zsym::Symbol
) where {T}
    # @show L
    M = Mp1 - 1 
    # M = (Int(sqrt(1 + 8L))-1)>>>1
    # Mp1 = M+1
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
    add_calc_logdet_expr!(q, M, T, logdetsym)
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
    if first(caches_logdiag(Mp1, LL, T)) # store log of diagonals
        lkj_length_triangle = VectorizationBase.align(StructuredMatrices.binomial2(Mp1+1), T)
        msym = gensym(:m)
        W = VectorizationBase.pick_vector_width(M, T)
        logdiag_quote = if Mp1 % W == 1
            quote
                $lkjsym[$(lkj_length_triangle + 1)] = zero($T)
                LoopVectorization.@vvectorize_unsafe $T for $msym ∈ 1:$M
                    $lkjsym[$msym+$(lkj_length_triangle+1)] = SLEEFPirates.log($lkjsym[$msym+1])
                end
            end
        else
            quote
                LoopVectorization.@vvectorize_unsafe $T for $msym ∈ 1:$(PaddedMatrices.calc_padding(Mp1,T))
                    $lkjsym[$msym+$lkj_length_triangle] = SLEEFPirates.log($lkjsym[$msym])
                end
            end
        end
        push!(q.args, macroexpand(LoopVectorization, logdiag_quote))
    end
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            $q
        end
        $logdetsym
    end
end

function lkj_adjoint_length(M)
    Ladj = 0
    for mp ∈ 1:M, mc ∈ mp+1:M+1
        Ladj += 1
    end
    for mc ∈ 2:M, mp ∈ 1:mc, mr ∈ mc+1:M+1
        Ladj += 1
    end
    Ladj
end

∂lkj_sym(i, j, k) = Symbol(:∂x_, i, :_, j, :_∂z_, k)

"""
This function also generates code to calculate the gradient with respect to the log determinant of the Jacobian,
as well as the full Jacobian of the constraining transformation.
"""
function constrain_lkj_factor_jac_quote(
    Mp1::Int, L::Int, LL::Int, ::Type{T}, lkjsym::Symbol, ∂logdetsym::Symbol, jacobiansym::Symbol, zsym::Symbol
) where {T}
    # @show L
    M = Mp1 - 1 
    # partial = true
    # Storage is the subdiagonal lower triangle
    # (the diagonal itself is determined by the correlation matrix constraint)
    #
    # The correlation matrix is (M+1) x (M+1)
    # our subdiagonal triangle is M x M
    #
    # We denote the (-1, 1) constrained parameters "z"
    # while "x" is the (M+1) x (M+1) cholesky factor of the correlation matrix.
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
    logdetsym = gensym(:logdet)
    add_calc_logdet_expr!(q, M, T, logdetsym)
    lkj_length_triangle = VectorizationBase.align(binomial2(Mp1+1),T)
    #lkj_length = lkj_length_triangle + VectorizationBase.align(Mp1, T)
    # lkjsym = gensym(:CorrCholesky)
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
    W = VectorizationBase.pick_vector_width(M, T)
    msym = gensym(:m)   
    logdiag_quote = if Mp1 % W == 1
        quote  # If the remainder is 1, we can save a loop iteration by simply assigning zero to the first element
            $lkjsym[$(1 + lkj_length_triangle)] = zero($T)
            LoopVectorization.@vvectorize_unsafe $T for $msym ∈ 1:$M
                $lkjsym[$msym + $(1+lkj_length_triangle)] = SLEEFPirates.log($lkjsym[$msym+1])
            end
        end
    else
        quote
            LoopVectorization.@vvectorize_unsafe $T for $msym ∈ 1:$(PaddedMatrices.calc_padding(Mp1, T))
                $lkjsym[$msym + $lkj_length_triangle] = SLEEFPirates.log($lkjsym[$msym])
            end
        end
    end
    push!(q.args, macroexpand(LoopVectorization, logdiag_quote))
    bin2M = binomial2(Mp1)
    i = 1
    for mc ∈ 1:M
        push!(q.args, :($∂logdetsym[$i] = zero($T)))
        i += 1
        for mr ∈ mc+2:M+1
            push!(q.args, :($∂logdetsym[$i] = $(Symbol(:∂ljp_, mr, :_, mc))))
            i += 1
        end
    end
    # Ladj = DistributionParameters.lkj_adjoint_length(M)
    # diagonal block of lkj
    i = 1
    for mp ∈ 1:M
        for mc ∈ mp+1:Mp1
            push!(q.args, :($jacobiansym[$i] = $(Symbol(:∂x_, mc, :_, mc, :_∂_, mp))))
            i +=1
        end
    end
    for mc ∈ 2:M
        for mp ∈ 1:mc
            for mr ∈ mc+1:Mp1 # iter through columns of z
                push!(q.args, :($jacobiansym[$i] = $(Symbol(:∂x_, mr, :_, mc, :_∂_, mp))))
                i += 1
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
    quote
        $(Expr(:meta,:inline))
        @fastmath @inbounds begin
            $q
        end
        $logdetsym
    end#, lkjsym, logdetsym, ∂logdetsym, jacobiansym
end

@generated function lkj_constrain!(
    lkjcorrchol::StructuredMatrices.AbstractMutableLowerTriangularMatrix{Mp1,T,LL},
    zlkj::PaddedMatrices.AbstractFixedSizeVector{L,T}
) where {T,L,Mp1,LL}
    constrain_lkj_factor_quote(Mp1, L, LL, T, :lkjcorrchol, :zlkj)
end
@generated function ∂lkj_constrain!(
    lkjcorrchol::StructuredMatrices.AbstractMutableLowerTriangularMatrix{Mp1,T,LL},
    lkjgrad::StructuredMatrices.AbstractMutableLowerTriangularMatrix{M,T},
    lkjjacobian::AbstractCholeskyConstraintAdjoint{Mp1,T,Ladj},
    zlkj::PaddedMatrices.AbstractFixedSizeVector{L,T}
) where {L,T,Mp1,LL,Ladj,M}
    constrain_lkj_factor_jac_quote(Mp1, L, LL, T, :lkjcorrchol, :lkjgrad, :lkjjacobian, :zlkj)
end


# @generated function lkj_constrain(
#     sp::StackPointer,
#     zlkj::PaddedMatrices.AbstractFixedSizeVector{L,T},
#     ::Val{Align} = Val{true}()
# ) where {L,T,Align}
#     lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym = constrain_lkj_factor_quote(L, T, :zlkj, true, Align)
#     quote
#         # Inlined because of:
#         # https://github.com/JuliaLang/julia/issues/32414
#         # Stop forcing inlining when the issue is fixed.
#         $(Expr(:meta,:inline))        
#         $lkjconstrain_q
#         sp, ($lkjconstrained_expr, $lkjlogdetsym)
#     end
# end
# @generated function ∂lkj_constrain(
#     sp::StackPointer,
#     zlkj::PaddedMatrices.AbstractFixedSizeVector{L,T}
# ) where {L,T}
#     lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym, lkjlogdetgrad, lkjjacobian = constrain_lkj_factor_jac_quote(L, T, :zlkj, true)
#     quote
#         # Inlined because of:
#         # https://github.com/JuliaLang/julia/issues/32414
#         # Stop forcing inlining when the issue is fixed.
#         $(Expr(:meta,:inline))
#         $lkjconstrain_q
#         sp, ($lkjconstrained_expr, $lkjlogdetsym, $lkjlogdetgrad, $lkjjacobian)
#     end
# end

function load_parameter!(
    first_pass::Vector{Any}, second_pass::Vector{Any}, out::Symbol,
    ::Type{<: AbstractCorrCholesky{M,T}}, partial::Bool,
    m::Module, sp::Union{Symbol,Nothing}, logjac::Bool = true, exportparam::Bool = false
) where {M,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    ninvlogitout = gensym(out)
    invlogitout = gensym(out)
    ∂invlogitout = gensym(out)
    use_sptr = sp isa Symbol
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    sumθᵢ = gensym(:sumθᵢ)
    N = (M * (M-1)) >>> 1
    Wm1 = W - 1
    rem = N & Wm1
    L = (N + Wm1) & ~Wm1
    zsym = gensym(:z) # z ∈ (-1, 1)
    zsymoffset = VectorizationBase.align(binomial2(M+1)*sizeof(T))
    if partial
        zsymoffset += VectorizationBase.align(max(2L,binomial2(M-1) + lkj_adjoint_length(M-1))*sizeof(T))
    end
    limitlife = use_sptr && !exportparam
    # zsym will be discarded, so we allocate it behind all the data we actually return.
    if use_sptr && !exportparam
        if partial
            push!(first_pass, :(($sp,$invlogitout) = $m.PtrVector{$L,$T}($sp)))
            push!(first_pass, :(($sp,$∂invlogitout) = $m.PtrVector{$L,$T}($sp)))
            limitlife && push!(first_pass, :($m.lifetime_start!($invlogitout); $m.lifetime_start!($∂invlogitout)))
        end
        # push!(first_pass, :( $zsym = $m.PtrVector{$N,$T}(pointer($sp + $zsymoffset,$T)) ))
        # push!(first_pass, :( ($sp,$zsym) = $m.PtrVector{$N,$T}($sp)))
        # limitlife && push!(first_pass, :($m.lifetime_start!($zsym)))
        push!(first_pass, :( $zsym = $m.PtrVector{$N,$T}(SIMDPirates.alloca(Val{$N}(),$T)) ))
    else
        push!(first_pass, :( $zsym = $m.PtrVector{$N,$T}(SIMDPirates.alloca(Val{$N}(),$T)) ))
        if partial
            push!(first_pass, :($invlogitout = $m.PtrVector{$L,$T}($m.alloca(Val{$L}(),$T))))
            push!(first_pass, :($∂invlogitout = $m.PtrVector{$L,$T}($m.alloca(Val{$L}(),$T))))
        end
    end
    i = gensym(:i)
    loop_body = quote $ninvlogitout = one($T) / (one($T) + $m.SLEEFPirates.exp($θ[$i])) end
    if partial
        push!(loop_body.args, :($invlogitout[$i] = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout[$i] = $ninvlogitout * $invlogitout[$i]))
        push!(loop_body.args, :($zsym[$i] = $m.SIMDPirates.vmuladd($(T(-2)), $ninvlogitout, one($T))))
        logjac && push!(loop_body.args, :(target = $m.vadd(target, $m.SLEEFPirates.log($∂invlogitout[$i]))))
    else
        push!(loop_body.args, :($invlogitout = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout = $ninvlogitout * $invlogitout))
        push!(loop_body.args, :($zsym[$i] = $m.SIMDPirates.vmuladd($(T(-2)), $ninvlogitout, one($T))))
        logjac && push!(loop_body.args, :(target = $m.vadd(target, $m.SLEEFPirates.log($∂invlogitout))))
    end
    vloop_quote = quote
        LoopVectorization.@vvectorize_unsafe $T $((m)) for $i ∈ 1:$N
            $loop_body
        end
    end
    push!(first_pass, macroexpand(m, vloop_quote))
    lkjlogdetsym = gensym(:lkjlogdetsym)
    if exportparam
        @assert partial == false
        lkj_length = binomial2(M+1)
    else
        lkj_length_triangle = VectorizationBase.align(binomial2(M+1),T)
        lkj_length = lkj_length_triangle + VectorizationBase.align(M, T)
    end
    if partial
        seedlkj = gensym(:seedlkj)
        lkjlogdetgradsym = gensym(:lkjlogdetgrad)
        lkjjacobiansym = gensym(:lkjjacobian)
        adjout = adj(out)
        push!(second_pass, :($m.muladd!($seedlkj, $adjout, $lkjjacobiansym)))
        loopq = quote
            LoopVectorization.@vvectorize_unsafe $T $((m)) for $i ∈ 1:$N
                $seedlkj[$i] = $(one(T)) - $(T(2)) * ( ($invlogitout)[$i] - (($seedlkj)[$i]) * ($∂invlogitout)[$i] )
            end
        end
        push!(second_pass, macroexpand(m, loopq))
        if limitlife
            for d ∈ (invlogitout, ∂invlogitout, adjout, out, lkjjacobiansym)
                push!(second_pass, :($m.lifetime_end!($d)))
            end
        end
        Ladj = DistributionParameters.lkj_adjoint_length(M - 1)
        if use_sptr #out
            push!(first_pass, :($lkjjacobiansym = $m.DistributionParameters.PtrCholeskyConstraintAdjoint{$M,$T,$Ladj}(pointer($sp,$T))))
            push!(first_pass, :($sp += $(VectorizationBase.align(Ladj*sizeof(T)))))
            push!(first_pass, :($out = $m.DistributionParameters.PtrCorrCholesky{$M,$T,$lkj_length}(pointer($sp,$T))))
            # push!(first_pass, Expr(:(=), out, :(PtrCorrCholesky{$M,$T,$lkj_length}(pointer($sp,$T)))))
            push!(first_pass, :($sp += $(VectorizationBase.align(sizeof(T)*lkj_length))))
        else
            push!(first_pass, :($lkjjacobiansym = $m.DistributionParameters.CholeskyConstraintAdjoint{$M,$T,$Ladj}(undef)))
            push!(first_pass, :($out = $m.DistributionParameters.CorrCholesky{$M,$T,$lkj_length}(undef)))
            # push!(first_pass, Expr(:(=), out, :(CorrCholesky{$M,$T,$lkj_length}(undef))))
        end
        push!(first_pass, :($seedlkj = $m.DistributionParameters.StructuredMatrices.PtrLowerTriangularMatrix{$(M-1),$T,$N}(pointer($∂θ))))
        push!(first_pass, :($lkjlogdetsym = $m.DistributionParameters.∂lkj_constrain!($out, $seedlkj, $lkjjacobiansym, $zsym)))
        push!(first_pass, :($adjout = $m.alloc_adjoint($out)))
        push!(first_pass, :($m.lifetime_start!($adjout)))
        push!(first_pass, :($θ += $N))
        push!(first_pass, :($∂θ += $N))
    else
        allocq = if use_sptr
            push!( first_pass, :( $out = $m.DistributionParameters.PtrCorrCholesky{$M,$T,$lkj_length}(pointer($sp,$T)) ) )
            # push!( first_pass, Expr(:(=), out, :(PtrCorrCholesky{$M,$T,$lkj_length}(pointer($sp,$T)) ) ))
            spoffset = sizeof(T)*lkj_length
            if !exportparam
                spoffset = VectorizationBase.align(spoffset)
            end
            push!( first_pass, :( $sp += $spoffset ) )
        else
            push!(first_pass, :( $out = $m.DistributionParameters.CorrCholesky{$M,$T,$lkj_length}(undef) ) )
            # push!(first_pass, Expr(:(=), out, :(CorrCholesky{$M,$T,$lkj_length}(undef) ) ))
        end
        push!(first_pass, :($lkjlogdetsym = $m.DistributionParameters.lkj_constrain!($out, $zsym)))
        push!(first_pass, :($θ += $N))
        limitlife && push!(second_pass, :($m.lifetime_end!($out)))
    end
    limitlife && push!(first_pass, :($m.lifetime_end!($zsym)))
    logjac && push!(first_pass, :(target = $m.vadd(target, $lkjlogdetsym)))
    nothing
end

function load_parameter!(
    first_pass::Vector{Any}, second_pass::Vector{Any}, out::Symbol,
    ::Type{<: AbstractCorrCholesky{M}}, partial::Bool = false,
    m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing,
    logjac::Bool = true, exportparam::Bool = false
) where {M}
    load_parameter!(first_pass, second_pass, out, CorrCholesky{M,Float64}, partial, m, sp, logjac, exportparam)
end


function parameter_names(::Type{<:StructuredMatrices.AbstractLowerTriangularMatrix{M}}, s::Symbol) where {M}
    ss = strip_hashtags(s)
    names = Vector{String}(undef, binomial2(M+1))
    for m ∈ 1:M
        names[m] = "$ss[$m,$m]"
    end
    ind = M
    for c ∈ 1:M-1, r ∈ c+1:M
        ind += 1
        names[ind] = "$ss[$r,$c]"
    end
    names::Vector{String}
end
parameter_names(x::LT) where {LT <: StructuredMatrices.AbstractLowerTriangularMatrix} = param_names(LT)

@generated function LinearAlgebra.mul!(
    σL::AbstractCovarCholesky{M,T,MTV},
    σ::Diagonal{T,<:AbstractFixedSizeVector{M,T,MV}},
    L::AbstractCorrCholesky{M,T,MTR}
) where {M,MTV,MTR,MV,T}
    q = quote
        σLpt = StructuredMatrices.PtrLowerTriangularMatrix{$M,Float64,$MTR}(pointer(σL))
        Lpt = StructuredMatrices.PtrLowerTriangularMatrix{$M,Float64,$MTR}(pointer(L))
        mul!(σLpt, σ, L)
    end
    σLcaches_ld = first(caches_logdiag(M, MTV, T))
    # tri_length = VectorizationBase.align(binomial2(M+1), T)
    # If σL does not have space, we don't calculate it.
    if !σLcaches_ld
        push!(q.args, :σL)
        return q
    end
    # We need at least one loop
    loopbody = quote
        logdiag_σL[m] = logσ[m] + logdiag_L[m]
    end
    # Do we also calculate invdiag?
    σLcaches_inv = first(caches_invdiag(M, MTV, T))
    if σLcaches_inv
        push!(q.args, :(invdiag_σL = invdiag(σL)))
        # push!(q.args, :(invdiag_L = invdiag(L)))
        push!(loopbody.args, :( invdiag_σL[m] = inv(σL[m]) ))
    end
    calc_logdiag_q = quote
        logdiag_σL = logdiag(σL)
        logdiag_L = logdiag(L)
        logσ = LazyMap(SLEEFPirates.log, σ.diag)
        @vvectorize $T for m ∈ 1:$MV
            $loopbody
        end
        σL
    end
    push!(q.args, calc_logdiag_q)
    q
end

@inline function Base.:*(
    sp::StackPointer,
    σ::Diagonal{T,<:AbstractFixedSizeVector{M,T,MV}},
    L::AbstractCorrCholesky{M,T,MTR}
) where {M,T,MV,MTR}
    (sp, σL) = PtrCovarCholesky{M,T}(sp)
    (sp, mul!(σL, σ, L))
end
    

