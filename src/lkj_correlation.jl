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
    L = VectorizationBase.align(VectorizationBase.align(VectorizationBase.align(binomial2(M+1), T) + M) + M)
    quote
        $(Expr(:meta,:inline))
        sp + $(L*sizeof(T)), PtrCovarCholesky{$M,$T,$L}(pointer(sp, T))
    end
end

@inline ReverseDiffExpressionsBase.alloc_adjoint(sp::StackPointer, C::AbstractCorrCholesky{M,T}) where {M,T} = StructuredMatrices.PtrLowerTriangularMatrix{M,T}(sp)
@inline ReverseDiffExpressionsBase.alloc_adjoint(sp::StackPointer, C::AbstractCovarCholesky{M,T}) where {M,T} = StructuredMatrices.PtrLowerTriangularMatrix{M,T}(sp)

function caches_logdiag(M, L, ::Type{T} = Float64) where {T}
    triangle_length = VectorizationBase.align(binomial2(M+1),T)
    L >= triangle_length + M
end
function caches_invdiag(M, L, ::Type{T} = Float64) where {T}
    triangle_length = VectorizationBase.align(VectorizationBase.align(binomial2(M+1),T) + M, T)
    triangle_length, L >= triangle_length + M
end
@inline logdiag(C::StructuredMatrices.AbstractDiagTriangularMatrix{M,T,L}) where {M,T,L} = PaddedMatrices.LazyMap(SLEEFPirates.log, PtrVector{M,T,M,true}(pointer(C)))
@inline invdiag(C::StructuredMatrices.AbstractDiagTriangularMatrix{M,T,L}) where {M,T,L} = PaddedMatrices.LazyMap(SIMDPirates.vinv, PtrVector{M,T,M,true}(pointer(C)))
@generated function logdiag(C::Union{<:AbstractCorrCholesky{M,T,L},<:AbstractCovarCholesky{M,T,L}}) where {M,T,L}
    if caches_logdiag(M,L)
        quote
            $(Expr(:meta,:inline))
            PtrVector{$M,$T,$(PaddedMatrices.calc_padding(M,T)),false}(pointer(C) + $(triangle_length*sizeof(T)))
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
    triangle_length, cid = caches_invdiag(M, L)
    if cid
        quote
            $(Expr(:meta,:inline))
            PtrVector{$M,$T,$(PaddedMatrices.calc_padding(M,T)),false}(pointer(C) + $(triangle_length*sizeof(T)))
        end
    else
        quote
            $(Expr(:meta,:inline))
            d = PtrVector{$M,$T,$M,true}(pointer(C))
            PaddedMatrices.LazyMap(SIMDPirates.vinv, d)
        end
    end
end


@generated PaddedMatrices.param_type_length(::Type{<: AbstractCorrCholesky{M}}) where {M} = StructuredMatrices.binomial2(M)
@generated PaddedMatrices.param_type_length(::AbstractCorrCholesky{M}) where {M} = StructuredMatrices.binomial2(M)

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
@inline VectorizationBase.vectorizable(A::PtrCorrCholesky{M,T}) where {M,T} = VectorizationBase.Pointer(A.ptr)
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

function corr_cholesky_adjoint_mul_quote(Mp1,T,sp::Bool = false)
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
    # for p ∈ 1:M
    #     push!(outtup.args, PaddedMatrices.sym(:∂lkj_∂z, p, p) )
    # end
    lkj_l = StructuredMatrices.binomial2(Mp1)
    lkj_l_full = PaddedMatrices.calc_padding(lkj_l, T)
    if sp
        push!(q.args, :(sptr = pointer(sp, $T); out = PtrVector{$lkj_l,$T}(sptr)))
        ind = 0
        size_T = sizeof(T)
        for pc ∈ 1:M
            for pr ∈ pc:M
                push!(q.args, :(VectorizationBase.store!(sptr + $size_T*$ind, $(PaddedMatrices.sym(:∂lkj_∂z, pr, pc)))))
                ind += 1
            end
        end
        push!(q.args, :(sp + $(VectorizationBase.align(lkj_l_full*size_T)), out' ))
    else
        outtup = Expr(:tuple,)
        for pc ∈ 1:M
            for pr ∈ pc:M
                push!(outtup.args, PaddedMatrices.sym(:∂lkj_∂z, pr, pc))
            end
        end
        for p ∈ StructuredMatrices.binomial2(Mp1)+1:lkj_l_full
            push!(outtup.args, zero(T))
        end
        push!(q.args, :(ConstantFixedSizeVector{$lkj_l, $T, $lkj_l_full}($outtup)'))
    end
    quote
        @fastmath @inbounds begin
            $q
        end
    end
end

@generated function Base.:*(
    t::LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractFixedSizeVector{BP,T,BPL}},
    adj::AbstractCholeskyConstraintAdjoint{Mp1,T,L}
) where {Mp1,T,L,BP,BPL}
    corr_cholesky_adjoint_mul_quote(Mp1,T)
end

@generated function Base.:*(
    t::StructuredMatrices.AbstractLowerTriangularMatrix{Mp1,T},
    adj::AbstractCholeskyConstraintAdjoint{Mp1,T,L}
) where {Mp1,T,L}
    corr_cholesky_adjoint_mul_quote(Mp1,T)
end

@generated function Base.:*(
    sp::StackPointer,
    t::LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractFixedSizeVector{BP,T,BPL}},
    adj::AbstractCholeskyConstraintAdjoint{Mp1,T,L}
) where {Mp1,T,L,BP,BPL}
    corr_cholesky_adjoint_mul_quote(Mp1,T,true)
end

@generated function Base.:*(
    sp::StackPointer,
    t::StructuredMatrices.AbstractLowerTriangularMatrix{Mp1,T},
    adj::AbstractCholeskyConstraintAdjoint{Mp1,T,L}
) where {Mp1,T,L}
    corr_cholesky_adjoint_mul_quote(Mp1,T,true)
end





"""
Generates the quote for the constraining transformation from z ∈ (-1,1) to _Correlation_Cholesky
without taking the derivative of the expression.
"""
function constrain_lkj_factor_quote(L::Int, T, zsym::Symbol, sp::Bool = false, align_sp::Bool = true)
    # @show L
    M = (Int(sqrt(1 + 8L))-1)>>>1
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
    if align_sp
        lkj_length_triangle = VectorizationBase.align(StructuredMatrices.binomial2(Mp1+1), T)
        lkj_length = lkj_length_triangle + VectorizationBase.align(Mp1, T)
    else
        lkj_length_triangle = StructuredMatrices.binomial2(Mp1+1)
        lkj_length = lkj_length_triangle
    end
    lkjsym = gensym(:CorrCholeksy)
    if sp
        push!(q.args, :($lkjsym = DistributionParameters.PtrCorrCholesky{$Mp1,$T,$lkj_length}(pointer(sp,$T))))
        sp_increment = VectorizationBase.align(sizeof(T) * lkj_length)
        push!(q.args, :(sp += $sp_increment))
    else
        push!(q.args, :($lkjsym = DistributionParameters.CorrCholesky{$Mp1,$T,$lkj_length}(undef)))
    end
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
    if align_sp # store log of diagonals
        msym = gensym(:m)
        W = VectorizationBase.pick_vector_width(M, T)
        logdiag_quote = if Mp1 % W == 1
            quote
                $lkjsym[$(lkj_length_triangle + 1)] = zero($T)
                @simd ivdep for $msym ∈ 1:$M
                    $lkjsym[$msym+$(lkj_length_triangle+1)] = SLEEFPirates.log($lkjsym[$msym+1])
                end
            end
        else
            quote
                @simd ivdep for $msym ∈ 1:$(PaddedMatrices.calc_padding(Mp1,T))
                    $lkjsym[$msym+$lkj_length_triangle] = SLEEFPirates.log($lkjsym[$msym])
                end
            end
        end
        push!(q.args, logdiag_quote)
    end
    return quote
        @fastmath @inbounds begin
            $q
        end
    end, lkjsym, logdetsym          
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
    M = (Int(sqrt(1 + 8L))-1)>>>1
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
    logdetsym = gensym(:logdet)
    if M > 1
        push!(q.args, :($logdetsym = Base.log( $(Expr(:call, :*, [Symbol(:ljp_,m) for m ∈ 3:Mp1]...)) ) ))
    else
        push!(q.args, :($logdetsym = zero(T)))
    end
    lkj_length_triangle = VectorizationBase.align(binomial2(Mp1+1),T)
    lkj_length = lkj_length_triangle + VectorizationBase.align(Mp1, T)
    lkjsym = gensym(:CorrCholesky)
    if sp
        push!(q.args, :($lkjsym = PtrCorrCholesky{$Mp1,$T,$lkj_length}(pointer(sp,$T))))
        push!(q.args, :(sp += $(VectorizationBase.align(sizeof(T)*lkj_length))))
    else
        push!(q.args, :($lkjsym = CorrCholesky{$Mp1,$T,$lkj_length}(undef)))
    end
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
    # If the remainder is 1, we can save a loop iteration by simply assigning zero to the first element
    # (lkjsym[1] == 1, so its log is 0)
        quote
            $lkjsym[$(1 + lkj_length_triangle)] = zero($T)
            @simd ivdep for $msym ∈ 1:$M
                $lkjsym[$msym + $(1+lkj_length_triangle)] = SLEEFPirates.log($lkjsym[$msym+1])
            end
        end
    else
        quote
            @simd ivdep for $msym ∈ 1:$(PaddedMatrices.calc_padding(Mp1, T))
                $lkjsym[$msym + $lkj_length_triangle] = SLEEFPirates.log($lkjsym[$msym])
            end
        end
    end
    push!(q.args, logdiag_quote)
    ∂logdetsym = gensym(:∂logdet)
    bin2M = binomial2(M+1)
    if sp
        push!(q.args, :($∂logdetsym = PtrVector{$bin2M,$T,$bin2M}(pointer(sp, $T))))
        push!(q.args, :(sp += $(VectorizationBase.align(sizeof(T)*bin2M))))
    else
        push!(q.args, :($∂logdetsym = FixedSizeVector{$bin2M,$T,$bin2M}(undef)))
    end
    i = 0
    for mc ∈ 1:M
        i += 1
        push!(q.args, :($∂logdetsym[$i] = zero($T)))
        for mr ∈ mc+2:M+1
            i += 1
            push!(q.args, :($∂logdetsym[$i] = $(Symbol(:∂ljp_, mr, :_, mc))))
        end
    end
    jacobiansym = gensym(:jacobian)
    Ladj = DistributionParameters.lkj_adjoint_length(M)
    if sp
        push!(q.args, :($jacobiansym = PtrCholeskyConstraintAdjoint{$Mp1,$T,$Ladj}(pointer(sp,$T))))
        push!(q.args, :(sp += $(VectorizationBase.align(Ladj*sizeof(T)))))
    else
        push!(q.args, :($jacobiansym = CholeskyConstraintAdjoint{$Mp1,$T,$Ladj}(undef)))
    end
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
end

@generated function lkj_constrain(zlkj::PaddedMatrices.AbstractFixedSizeVector{L,T}) where {T,L}
    lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym = constrain_lkj_factor_quote(L, T, :zlkj)
    quote
        $lkjconstrain_q
        $lkjconstrained_expr, $lkjlogdetsym
    end
end
@generated function ∂lkj_constrain(zlkj::PaddedMatrices.AbstractFixedSizeVector{L,T}) where {L,T}
    lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym, lkjlogdetgrad, lkjjacobian = constrain_lkj_factor_jac_quote(L, T, :zlkj)
    quote
        $lkjconstrain_q
        $lkjconstrained_expr, $lkjlogdetsym, $lkjlogdetgrad, $lkjjacobian
    end
end


@generated function lkj_constrain(sp::StackPointer, zlkj::PaddedMatrices.AbstractFixedSizeVector{L,T}, ::Val{Align} = Val{true}()) where {L,T,Align}
    lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym = constrain_lkj_factor_quote(L, T, :zlkj, true, Align)
    quote
        # Inlined because of:
        # https://github.com/JuliaLang/julia/issues/32414
        # Stop forcing inlining when the issue is fixed.
        $(Expr(:meta,:inline))        
        $lkjconstrain_q
        sp, ($lkjconstrained_expr, $lkjlogdetsym)
    end
end
@generated function ∂lkj_constrain(sp::StackPointer, zlkj::PaddedMatrices.AbstractFixedSizeVector{L,T}) where {L,T}
    lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym, lkjlogdetgrad, lkjjacobian = constrain_lkj_factor_jac_quote(L, T, :zlkj, true)
    quote
        # Inlined because of:
        # https://github.com/JuliaLang/julia/issues/32414
        # Stop forcing inlining when the issue is fixed.
        $(Expr(:meta,:inline))
        $lkjconstrain_q
        sp, ($lkjconstrained_expr, $lkjlogdetsym, $lkjlogdetgrad, $lkjjacobian)
    end
end



function load_parameter!(
    first_pass, second_pass, out, ::Type{<: AbstractCorrCholesky{M,T}},
    partial::Bool = false, m::Module = DistributionParameters, sp::Nothing = nothing,
    logjac::Bool = true, exportparam::Bool = false
) where {M,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    ninvlogitout = gensym(out)
    invlogitout = gensym(out)
    ∂invlogitout = gensym(out)

    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    sumθᵢ = gensym(:sumθᵢ)

    N = (M * (M-1)) >>> 1

    Wm1 = W - 1
    rem = N & Wm1
    L = (N + Wm1) & ~Wm1
    # @show N, L
    log_jac = gensym(:log_jac)
    zsym = gensym(:z) # z ∈ (-1, 1)
    q = quote
        $zsym = FixedSizeVector{$N,$T}(undef)
        # $log_jac = DistributionParameters.SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T))
        $log_jac = zero($T)
    end
    if partial
        push!(q.args, :($invlogitout = FixedSizeVector{$L,$T}(undef)))
        push!(q.args, :($∂invlogitout = FixedSizeVector{$L,$T}(undef)))
    end
    i = gensym(:i)
    loop_body = quote
        # $ninvlogitout = one($T) / (one($T) + $m.SLEEFPirates.exp($(T(0.5)) * $θ[$i]))
        $ninvlogitout = one($T) / (one($T) + $m.SLEEFPirates.exp($θ[$i]))
    end
    if partial
        push!(loop_body.args, :($invlogitout[$i] = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout[$i] = $ninvlogitout * $invlogitout[$i]))
        push!(loop_body.args, :($zsym[$i] = $m.SIMDPirates.vmuladd($(T(-2)), $ninvlogitout, one($T))))
        push!(loop_body.args, :($log_jac += $m.SLEEFPirates.log($∂invlogitout[$i])))
    else
        push!(loop_body.args, :($invlogitout = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout = $ninvlogitout * $invlogitout))
        push!(loop_body.args, :($zsym[$i] = $m.SIMDPirates.vmuladd($(T(-2)), $ninvlogitout, one($T))))
        push!(loop_body.args, :($log_jac += $m.SLEEFPirates.log($∂invlogitout)))
    end

    push!(q.args, macroexpand(m, quote
        LoopVectorization.@vvectorize $T $((m)) for $i ∈ 1:$N
            $loop_body
        end
    end))
    lkjlogdetsym = gensym(:lkjlogdetsym)
    if partial
        # lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym, lkjlogdetgrad, lkjjacobian = constrain_lkj_factor_jac_quote(N, T, zsym)
        seedlkj = gensym(:seedlkj)
        lkjlogdetgradsym = gensym(:lkjlogdetgrad)
        lkjjacobiansym = gensym(:lkjjacobian)
        spq = quote
            $seedlkj = ($(adj(out)) * $lkjjacobiansym).parent
            LoopVectorization.@vvectorize $T $((m)) for $i ∈ 1:$L
                $seedlkj[$i] = $(one(T)) - $(T(2)) * ( ($invlogitout)[$i] - (($seedlkj)[$i] + $lkjlogdetgradsym[$i]) * ($∂invlogitout)[$i] )
            end
        end
        push!(second_pass, macroexpand(m, spq))
        push!(q.args, quote
            # $zsym = ConstantFixedSizeVector{$M}($mv)
            # $lkjconstrain_q
            # $out = $lkjconstrained_expr
              $out, $lkjlogdetsym, $lkjlogdetgradsym, $lkjjacobiansym = DistributionParameters.∂lkj_constrain($zsym)
              $θ += $N
              $seedlkj = ReverseDiffExpressionsBase.alloc_adjoint(pointer(∂θ), $out)
              # $seedlkj = PtrVector{$N,$T,$N,true}(pointer(∂θ))
              $∂θ += $N
            # target += DistributionParameters.SIMDPirates.vsum($log_jac) + $lkjlogdetsym
            # target += $(T(0.5)) * ($log_jac + $lkjlogdetsym)
            # println("log_jac invlogit")
            # println($log_jac)
            # println("log_jac lkj")
            # println($lkjlogdetsym)
        end)
        # @show second_pass
    else
        # lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym = constrain_lkj_factor_quote(N, T, zsym)

        push!(q.args, quote
            # $zsym = ConstantFixedSizeVector{$M}($mv)
            # $lkjconstrain_q
            # $out = $lkjconstrained_expr
            $out, $lkjlogdetsym = DistributionParameters.lkj_constrain($zsym)
            $θ += $N
            # target += DistributionParameters.SIMDPirates.vsum($log_jac) + $lkjlogdetsym
            
#            target += ($log_jac + $lkjlogdetsym)
        end)
    end
    logjac && push!(q.args, :(target = $m.vadd(target, $log_jac + $lkjlogdetsym)))
    # push!(q.args, :(@show $zsym))

    push!(first_pass, q)

    nothing
end
function load_parameter!(
    first_pass, second_pass, out, ::Type{<: AbstractCorrCholesky{M,T}},
    partial::Bool, m::Module, sp::Symbol, logjac::Bool = true, exportparam::Bool = false
) where {M,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    ninvlogitout = gensym(out)
    invlogitout = gensym(out)
    ∂invlogitout = gensym(out)

    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    sumθᵢ = gensym(:sumθᵢ)

    N = (M * (M-1)) >>> 1

    Wm1 = W - 1
    rem = N & Wm1
    L = (N + Wm1) & ~Wm1
    # @show N, L
 #   if logjac
 #       log_jac = gensym(:log_jac)
 #   end
    zsym = gensym(:z) # z ∈ (-1, 1)
    zsymoffset = binomial2(M+1)
    if partial
        zsymoffset += VectorizationBase.align(max(2L,binomial2(M-1) + lkj_adjoint_length(M-1)),T)
    end
    # zsym will be discarded, so we allocate it behind all the data we actually return.
    q = quote
        $zsym = $m.PtrVector{$N,$T}(pointer($sp + $(zsymoffset*sizeof(T)),$T))
#        $zsym = FixedSizeVector{$N,$T}(undef)
        # $log_jac = DistributionParameters.SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T))
    end
#    logjac && push!(q.args, Expr(:(=), log_jac, zero(T)))
    if partial
        push!(q.args, :(($sp,$invlogitout) = $m.PtrVector{$L,$T}($sp)))
        push!(q.args, :(($sp,$∂invlogitout) = $m.PtrVector{$L,$T}($sp)))
    end
    i = gensym(:i)
    loop_body = quote
        # $ninvlogitout = one($T) / (one($T) + SLEEFPirates.exp($(T(0.5)) * $θ[$i]))
        $ninvlogitout = one($T) / (one($T) + $m.SLEEFPirates.exp($θ[$i]))
    end
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
        LoopVectorization.@vvectorize $T $((m)) for $i ∈ 1:$N
            $loop_body
        end
y    end
    #    println("\n\n\n\n\n")
#    println(vloop)
#    println("\n\n\n\n\n")
    push!(q.args, macroexpand(m, vloop_quote))
    lkjlogdetsym = gensym(:lkjlogdetsym)
    if partial
        # lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym, lkjlogdetgrad, lkjjacobian = constrain_lkj_factor_jac_quote(N, T, zsym)
        seedlkj = gensym(:seedlkj)
        lkjlogdetgradsym = gensym(:lkjlogdetgrad)
        lkjjacobiansym = gensym(:lkjjacobian)
        seedlkjgensym = gensym(seedlkj)
        push!(second_pass, quote
              ($sp, $seedlkjgensym) = $sp * $(adj(out)) * $lkjjacobiansym
              $seedlkj = $seedlkjgensym.parent
              $(macroexpand(m, quote
                            LoopVectorization.@vvectorize $T $((m)) for $i ∈ 1:$N
                            $∂θ[$i] = $(one(T)) - $(T(2)) * ( ($invlogitout)[$i] - (($seedlkj)[$i] + $lkjlogdetgradsym[$i]) * ($∂invlogitout)[$i] )
                            end
                            end))
            $∂θ += $N
        end)
        push!(q.args, quote
            ($sp, ($out, $lkjlogdetsym, $lkjlogdetgradsym, $lkjjacobiansym)) = DistributionParameters.∂lkj_constrain($sp, $zsym)
            $θ += $N
            # target += DistributionParameters.SIMDPirates.vsum($log_jac) + $lkjlogdetsym
            # target += $(T(0.5)) * ($log_jac + $lkjlogdetsym)
            # println("log_jac invlogit")
            # println($log_jac)
            # println("log_jac lkj")
            # println($lkjlogdetsym)

#            target += ($log_jac +  $lkjlogdetsym)
              end)
        # @show second_pass
    else
        # lkjconstrain_q, lkjconstrained_expr, lkjlogdetsym = constrain_lkj_factor_quote(N, T, zsym)

        push!(q.args, quote
            ($sp, ($out, $lkjlogdetsym)) = DistributionParameters.lkj_constrain($sp, $zsym, Val{$(!exportparam)}())
            $θ += $N
            # target += DistributionParameters.SIMDPirates.vsum($log_jac) + $lkjlogdetsym
#            target = $m.SIMDPirates.$m.vadd(target, $log_jac + $lkjlogdetsym)
#            target += ($log_jac + $lkjlogdetsym)
        end)
    end
    logjac && push!(q.args, :( target = $m.vadd(target, $lkjlogdetsym)))

    # push!(q.args, :(@show $zsym))

    push!(first_pass, q)

    nothing
end

function load_parameter!(
    first_pass, second_pass, out, ::Type{<: AbstractCorrCholesky{M}},
    partial::Bool = false, m::Module = DistributionParameters,
    sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true, exportparam::Bool = false
) where {M}
    load_parameter!(first_pass, second_pass, out, CorrCholesky{M,Float64}, partial, m, sp, logjac, exportparam)
end


function parameter_names(::Type{<: AbstractCorrCholesky{M}}, s::Symbol) where {M}
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
    tri_length = VectorizationBase.align(binomial2(M+1), T)
    # If it is too small to cache logdiag, don't.
    if MTR < triangle_length + M
        push!(q.args, :σL)
        return q
    end
    loopbody = quote
        logdiag_σL[m] = logσ[m] + logdiag_L[m]
    end
    if MTV >= MTR + M
        push!(q.args, :(invdiag_σL = invdiag(σL)))
        push!(loopbody.args, :( invdiag_σL[m] = vinv(σL[m]) ))
    end
    calc_logdiag_q = quote
        logdiag_σL = logdiag(σL)
        logdiag_L = logdiag(L)
        logσ = LazyMap(SLEEFPirates.log, σ)
        @vvectorize for m ∈ 1:$MV
            $loopbody
        end
        σL
    end
    push!(q.args, calc_logdiag_q)
    q
end

function Base.:*(
    sp::StackPointer,
    σ::Diagonal{T,<:AbstractFixedSizeVector{M,T,MV}},
    L::AbstractCorrCholesky{M,T,MTR}
) where {M,T,MV,MTR}
    (sp, σL) = PtrCovarCholesky{M,T}(sp)
    (sp, mul!(σL, σ, L))
end
    

