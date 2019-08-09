module DistributionParameters

using PaddedMatrices, StructuredMatrices, LinearAlgebra,
        SIMDPirates, SLEEFPirates, LoopVectorization, VectorizationBase

using PaddedMatrices: StackPointer, DynamicPtrMatrix,
    AbstractMutableFixedSizePaddedVector, AbstractFixedSizePaddedVector
export RealFloat, RealArray, RealVector, RealMatrix, Bounds, MissingDataArray, maybe_missing,
    constrain, LKJCorrCholesky, DynamicCovarianceMatrix, Target

#using LoopVectorization: @vvectorize
import SIMDPirates: vsum, vadd, vifelse

struct One end
# This should be the only method I have to define.
@inline Base.:*(::One, a) = a
#@inline Base.:*(a, ::One) = a
@inline Base.:*(a::StackPointer, b::One) = (a, b)
# But I'll define this one too. Would it be better not to, so that we get errors
# if the seed is for some reason multiplied on the right?
#@inline Base.:*(::One, ::One) = One()
Base.size(::One) = Core.tuple()


struct Target{T,W}
    v::Vec{W,T}
    s::T
end
function vsum(t::Target)
    Base.FastMath.add_fast(SIMDPirates.vsum(t.v), t.s)
end
#
@inline function vadd(v::Vec{W,T}, t::Target{T,W}) where {W,T}
    Target( SIMDPirates.vadd(v, t.v), t.s )
end
@inline function vadd(v::SVec{W,T}, t::Target{T,W}) where {W,T}
    Target( SIMDPirates.vadd(SIMDPirates.extract_data(v), t.v), t.s )
end
@inline function vadd(t::Target{T,W}, v::Vec{W,T}) where {W,T}
    Target( SIMDPirates.vadd(v, t.v), t.s )
end
@inline function vadd(t::Target{T,W}, v::SVec{W,T}) where {W,T}
    Target( SIMDPirates.vadd(SIMDPirates.extract_data(v), t.v), t.s )
end
#=
@inline function add(v::Vec{W1,T}, t::Target{T,W2}) where {W1,W2,T}
    Target( t.v, Base.FastMath.add_fast(SIMDPirates.vsum(v), t.s) )
end
=#
@generated function vadd(t::Target{T,W1}, v::Vec{W2,T}) where {W1,W2,T}
    if W1 > W2
        largerv = :(t.v)
        littlev = :(v)
        Ws, Wl = W2, W1
    else
        largerv = :(v)
        littlev = :(t.v)
        Ws, Wl = W1, W2
    end
    quote
        $(Expr(:meta,:inline))
        vs = $littlev
        vb = $largerv
        vse = $(Expr(:tuple,[:(vs[$w]) for w ∈ 1:Ws]...,[Core.VecElement(zero(T)) for _ ∈ Ws+1:W2]...))
        Target(
            SIMDPirates.vadd(vb, vse),
            t.s
        )
    end
end
@generated function vadd(v::Vec{W2,T}, t::Target{T,W1}) where {W1,W2,T}
    if W1 > W2
        largerv = :(t.v)
        littlev = :(v)
        Ws, Wl = W2, W1
    else
        largerv = :(v)
        littlev = :(t.v)
        Ws, Wl = W1, W2
    end
    quote
        $(Expr(:meta,:inline))
        vs = $littlev
        vb = $largerv
        vse = $(Expr(:tuple,[:(vs[$w]) for w ∈ 1:Ws]...,[Core.VecElement(zero(T)) for _ ∈ Ws+1:W2]...))
        Target(
            SIMDPirates.vadd(vb, vse),
            t.s
        )
    end
end
@inline vadd(t::Target, v::SVec) = add(t, SIMDPirates.extract_data(v))
@inline vadd(v::SVec, t::Target) = add(t, SIMDPirates.extract_data(v))
@inline function vadd(s::T, t::Target{T,W}) where {W,T}
    Target( t.v, Base.FastMath.add_fast(s, t.s) )
end
@inline function vadd(t::Target{T,W}, s::T) where {W,T}
    Target( t.v, Base.FastMath.add_fast(s, t.s) )
end
@inline function vadd(t1::Target{T,W}, t2::Target{T,W}) where {W,T}
    Target(
        SIMDPirates.vadd(t1.v, t2.v),
        Base.FastMath.add_fast(t1.s, t2.s)
    )
end
@generated function vadd(t1::Target{T,W1}, t2::Target{T,W2}) where {W1,W2,T}
    if W1 > W2
        largerv = :(t1.v)
        littlev = :(t2.v)
        Ws, Wl = W2, W1
    else
        largerv = :(t2.v)
        littlev = :(t1.v)
        Ws, Wl = W1, W2
    end
    quote
        $(Expr(:meta,:inline))
        vs = $littlev
        vb = $largerv
        vse = $(Expr(:tuple,[:(vs[$w]) for w ∈ 1:Ws]...,[Core.VecElement(zero(T)) for _ ∈ Ws+1:W2]...))
        Target(
            SIMDPirates.vadd(vb, vse),
            Base.FastMath.add_fast(t1.s, t2.s)#, SIMDPirates.vsum($(W1 > W2 ? :(t2.v) : :(t1.v))))
        )
    end
end
#
#=
@inline function vadd(v::Vec{W,T}, t::Target{T,W}) where {W,T}
    @show Target( SIMDPirates.vadd(v, t.v), t.s )
end
@inline function vadd(v::SVec{W,T}, t::Target{T,W}) where {W,T}
    @show Target( SIMDPirates.vadd(SIMDPirates.extract_data(v), t.v), t.s )
end
@inline function vadd(t::Target{T,W}, v::Vec{W,T}) where {W,T}
    @show Target( SIMDPirates.vadd(v, t.v), t.s )
end
@inline function vadd(t::Target{T,W}, v::SVec{W,T}) where {W,T}
    @show Target( SIMDPirates.vadd(SIMDPirates.extract_data(v), t.v), t.s )
end
#=
@inline function add(v::Vec{W1,T}, t::Target{T,W2}) where {W1,W2,T}
    Target( t.v, Base.FastMath.add_fast(SIMDPirates.vsum(v), t.s) )
end
=#
@generated function vadd(t::Target{T,W1}, v::Vec{W2,T}) where {W1,W2,T}
    if W1 > W2
        largerv = :(t.v)
        littlev = :(v)
        Ws, Wl = W2, W1
    else
        largerv = :(v)
        littlev = :(t.v)
        Ws, Wl = W1, W2
    end
    quote
        $(Expr(:meta,:inline))
        vs = $littlev
        vb = $largerv
        vse = $(Expr(:tuple,[:(vs[$w]) for w ∈ 1:Ws]...,[Core.VecElement(zero(T)) for _ ∈ Ws+1:W2]...))
        @show Target(
            SIMDPirates.vadd(vb, vse),
            t.s
        )
    end
end
@generated function vadd(v::Vec{W2,T}, t::Target{T,W1}) where {W1,W2,T}
    if W1 > W2
        largerv = :(t.v)
        littlev = :(v)
        Ws, Wl = W2, W1
    else
        largerv = :(v)
        littlev = :(t.v)
        Ws, Wl = W1, W2
    end
    quote
        $(Expr(:meta,:inline))
        vs = $littlev
        vb = $largerv
        vse = $(Expr(:tuple,[:(vs[$w]) for w ∈ 1:Ws]...,[Core.VecElement(zero(T)) for _ ∈ Ws+1:W2]...))
        @show Target(
            SIMDPirates.vadd(vb, vse),
            t.s
        )
    end
end
@inline vadd(t::Target, v::SVec) = @show add(t, SIMDPirates.extract_data(v))
@inline vadd(v::SVec, t::Target) = @show add(t, SIMDPirates.extract_data(v))
@inline function vadd(s::T, t::Target{T,W}) where {W,T}
    @show Target( t.v, Base.FastMath.add_fast(s, t.s) )
end
@inline function vadd(t::Target{T,W}, s::T) where {W,T}
    @show Target( t.v, Base.FastMath.add_fast(s, t.s) )
end
@inline function vadd(t1::Target{T,W}, t2::Target{T,W}) where {W,T}
    @show Target(
        SIMDPirates.vadd(t1.v, t2.v),
        Base.FastMath.add_fast(t1.s, t2.s)
    )
end
@generated function vadd(t1::Target{T,W1}, t2::Target{T,W2}) where {W1,W2,T}
    if W1 > W2
        largerv = :(t1.v)
        littlev = :(t2.v)
        Ws, Wl = W2, W1
    else
        largerv = :(t2.v)
        littlev = :(t1.v)
        Ws, Wl = W1, W2
    end
    quote
        $(Expr(:meta,:inline))
        vs = $littlev
        vb = $largerv
        vse = $(Expr(:tuple,[:(vs[$w]) for w ∈ 1:Ws]...,[Core.VecElement(zero(T)) for _ ∈ Ws+1:W2]...))
        @show Target(
            SIMDPirates.vadd(vb, vse),
            Base.FastMath.add_fast(t1.s, t2.s)#, SIMDPirates.vsum($(W1 > W2 ? :(t2.v) : :(t1.v))))
        )
    end
end

=#
@inline function vifelse(mask::Union{Vec{W,Bool},<:Unsigned}, t1::Target{T,W}, t2::Target{T,W}) where {W,T}
    Target(
        vifelse(mask, t1.v, t2.v), t2.s
    )
end

#@generated function Base.zero(::Type{Target{T}}) where {T}
# API deliberately does not support zero.
@generated function initialize_target(::Type{T}) where {T}
    W = VectorizationBase.pick_vector_width(T)
    quote
        $(Expr(:meta,:inline))
        Target(
            SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T)),
            zero($T)
        )
    end
end


"""
Transform a parameter vector to the constrained space.
"""
function constrain end
function constrain! end
function constrained_length end
function parameter_names end

include("constraints.jl")
include("uniform_mapped_parameters.jl")
include("lkj_correlation.jl")
include("normal_variates.jl")
include("missing_data.jl")
# include("autoregressive_matrix.jl")

# @support_stack_pointer load_parameter

PaddedMatrices.@support_stack_pointer lkj_constrain
PaddedMatrices.@support_stack_pointer ∂lkj_constrain
PaddedMatrices.@support_stack_pointer CovarianceMatrix
PaddedMatrices.@support_stack_pointer ∂CovarianceMatrix
function __init__()
    for m ∈ (:lkj_constrain, :∂lkj_constrain, :CovarianceMatrix, :∂CovarianceMatrix)
        push!(PaddedMatrices.STACK_POINTER_SUPPORTED_METHODS, m)
    end
end

end # module
