
# API requires defining 4 methods for a constraint:
# 1. Either of the following, returning constrained:
#    a) constrain(::Ptr, ::constraint)
#    b) stack_pointer_call(::typeof(constrain), ::StackPointer, ::Ptr, ::constraint)
# 2. Either of the following, returning (constrained, ∂):
#    a) constrain_pullback(::Ptr, ::constraint)
#    b) stack_pointer_call(::typeof(constrain_pullback), ::StackPointer, ::Ptr, ::constraint)
# 3. Either of the following, returning (adj):
#    a) alloc_adj(∇::Ptr, ::constraint)
#    b) stack_pointer_call(::typeof(alloc_adj), ::StackPointer, ::Ptr, ::constraint)
# 4. constrain_reverse!(adj, ∂, ::constraint)
#    (alloc_adj may return a special data structure holding a stack pointer and the appropriate piece from ∇ to be used in constrain_reverse!)


# I shouldn't need the following three fallbacks?!?!
@generated function index_description(::Val{descript}, ::Val{I}) where {descript, I}
    Expr(:call, Expr(:curly, :Val, descript[I]))
end
function stack_pointer_call(::typeof(constrain), sp::StackPointer, θ::Ptr{Float64}, ::Val{descript}, ::Val{I}) where {descript, I}
    θ = gep(θ, parameter_offset(Val{descript}(), Val{I}()))
    stack_pointer_call(constrain, sp, θ, index_description(Val{descript}(), Val{I}()))
end
function constrain(θ::Ptr{Float64}, ::Val{descript}, ::Val{I}) where {descript, I}
    θ = gep(θ, parameter_offset(Val{descript}(), Val{I}()))
    constrain(θ, index_description(Val{descript}(), Val{I}()))
end

constrain(θ::Ptr{Float64}, ::RealScalar{-Inf,Inf}) = (Zero(), load(θ))
constrain(θ::Ptr{Float64}, ::RealScalar{0.0,Inf}) = (x = load(θ); (x, exp(x)))
constrain(θ::Ptr{Float64}, ::RealScalar{Inf,0.0}) = (x = load(θ); (x, Base.FastMath.sub_fast(exp(x))))
constrain(θ::Ptr{Float64}, ::RealScalar{L,Inf}) where {L} = (x = load(θ); (x, Base.FastMath.add_fast(L, exp(x))))
constrain(θ::Ptr{Float64}, ::RealScalar{Inf,U}) where {U} = (x = load(θ); (x, Base.FastMath.sub_fast(U, exp(x))))
constrain(θ::Ptr{Float64}, ::RealScalar{0.0,1.0}) = (x = ninvlogit(load(θ)); (muladd(Base.FastMath.sub_fast(x),x,x), x))
# constrain(θ::Ptr{Float64}, ::RealScalar{L,U}) where {L,U} = (x = ninvlogit(load(θ)); (muladd(Base.FastMath.sub_fast(x),x,x), x))
constrain_pullback(θ::Ptr{Float64}, ::RealScalar{-Inf,Inf}) = (Zero(), load(θ), ReverseDiffExpressionBase.One())
function constrain_pullback(θ::Ptr{Float64}, ::RealScalar{0.0,Inf})
    y = load(θ)
    x = exp(y)
    (y, x, x)
end
function constrain_pullback(θ::Ptr{Float64}, ::RealScalar{Inf,0.0})
    y = load(θ)
    x = Base.FastMath.sub_fast(exp(y))
    (y, x, x)
end
function constrain_pullback(θ::Ptr{Float64}, ::RealScalar{L,Inf}) where {L}
    y = load(θ)
    x = exp(y)
    (y, Base.FastMath.add_fast(L, x), x)
end
function constrain_pullback(θ::Ptr{Float64}, ::RealScalar{Inf,U}) where {U}
    y = load(θ)
    x = Base.FastMath.sub_fast(exp(y))
    (y, Base.FastMath.add_fast(U, x), x)
end
# constrain_pullback(θ::Ptr{Float64}, ::RealScalar{0.0,1.0}) = (x = ninvlogit(load(θ)); (x, Base.FastMath.mul_fast(x, Base.FastMath.sub_fast(x, 1.0))))
alloc_adj(::Any, ::RealScalar) = Zero()
constrain_reverse!(∇::Ptr{T}, adj::T, ∂::One, ::RealScalar) where {T} = store!(∇, adj)
constrain_reverse!(∇::Ptr, adj, ∂::One, ::RealScalar) = store!(∇, vsum(adj))
constrain_reverse!(∇::Ptr{T}, adj::T, ∂, ::RealScalar) where {T} = store!(∇, vmul(adj, ∂))
constrain_reverse!(∇::Ptr, adj, ∂, ::RealScalar) = store!(∇, vsum(vmul(adj, ∂)))

@inline alloc_adj(∇::Ptr{Float64}, ::RealArray{S,<:Any,<:Any,0}) where {S} = NoPadPtrView{S}(∇)
@inline constrain(θ::Ptr{Float64}, ::RealArray{S,-Inf,Inf,0}) where {S} = NoPadPtrView{S}(θ)
@inline constrain_pullback(θ::Ptr{Float64}, ::RealArray{S,-Inf,Inf,0}) where {S} = (NoPadPtrView{S}(θ), ReverseDiffExpressionBase.One())
# @inline alloc_adj(∇::Ptr{Float64}, ::RealArray{S,-Inf,Inf,0}) where {S} = NoPadPtrView{S}(∇)
@inline constrain_reverse!(adj::PtrArray{S}, ∂::One, ::RealArray{S,-Inf,Inf}) = nothing

# @inline constrain(θ::Ptr{Float64}, ::RealArray{S,0.0,Inf,0}) where {S} = NoPadPtrView{S}(θ)
# @inline constrain_pullback(θ::Ptr{Float64}, ::RealArray{S,0.0,Inf,0}) where {S} = (NoPadPtrView{S}(θ), ReverseDiffExpressionBase.One())
# @inline alloc_adj(∇::Ptr{Float64}, ::RealArray{S,0.0,Inf}) where {S} = NoPadPtrView{S}(∇)
# @inline constrain_reverse!(∇::Ptr{Float64}, adj::PtrArray{S}, ∂::One, ::RealArray{S,0.0,Inf}) = nothing

@inline function constrain_lower_bounded_array(sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,0.0,Inf,0}) where {S}
    sp, e = PtrArray{X}(sp)
    ev = flatvector(e)
    uv = flatvector(NoPadPtrView{S}(θ))
    t = vzero()
    @avx for i ∈ eachindex(uv)
        euv = exp(uv[i])
        t += euv
        ev[i] = euv
    end
    sp, t, e, uv
end
@inline function constrain_lower_bounded_array(sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,L,Inf,0}) where {L,S}
    sp, e = PtrArray{X}(sp)
    ev = flatvector(e)
    uv = flatvector(NoPadPtrView{S}(θ))
    t = vzero()
    @avx for i ∈ eachindex(uv)
        euv = exp(uv[i])
        t += euv
        ev[i] = L + euv
    end
    sp, t, e, uv
end
@inline function stack_pointer_call(::typeof(constrain), sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,L,Inf,0}) where {L,S}
    sp, t, e, uv = constrain_lower_bounded_array(sp, θ, RealArray{S,L,Inf,0}())
    sp, (t, MappedArray(e, pointer(uv)))
end
@inline function stack_pointer_call(::typeof(constrain_pullback), sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,L,Inf,0}) where {L,S}
    sp, t, e, uv = constrain_lower_bounded_array(sp, θ, RealArray{S,L,Inf,0}())
    sp, (t, MappedArray(e, pointer(uv)), e)
end
# @inline alloc_adj(∇::Ptr{Float64}, ::RealArray{S,-Inf,Inf,0}) where {S} = NoPadPtrView{S}(∇)
# function stack_pointer_call(::typeof(alloc_adj), sp::StackPointer, θ::Ptr{Float64}
function constrain_reverse!(adj::PtrArray{S}, ∂::PtrArray{S}, ::RealArray{S,0.0,Inf}) where {S}
    va = flatvector(adj)
    v∂ = flatvector(∂)
    @avx for i ∈ eachindex(va)
        va[i] = va[i] * v∂[i] + 1.0
    end
end
function constrain_reverse!(adj::PtrArray{S}, ∂::PtrArray{S}, ::RealArray{S,L,Inf}) where {L,S}
    va = flatvector(adj)
    v∂ = flatvector(∂)
    @avx for i ∈ eachindex(va)
        va[i] = va[i] * (v∂[i]-L) + 1.0
    end
end


@inline function stack_pointer_call(::typeof(constrain), sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,Inf,L,0}) where {S}
    sp, t, e, uv = constrain_positive_array(sp, θ, RealArray{S,0.0,Inf,0}())
    sp, (t, MappedArray(e, pointer(uv)))
end
@inline function stack_pointer_call(::typeof(constrain_pullback), sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,Inf,L,0}) where {S}
    sp, t, e, uv = constrain_positive_array(sp, θ, RealArray{S,0.0,Inf,0}())
    sp, (t, MappedArray(e, pointer(uv)), e)
end
# @inline alloc_adj(∇::Ptr{Float64}, ::RealArray{S,-Inf,Inf,0}) where {S} = NoPadPtrView{S}(∇)
# function stack_pointer_call(::typeof(alloc_adj), sp::StackPointer, θ::Ptr{Float64}
function constrain_reverse!(adj::PtrArray{S}, ∂::PtrArray{S}, ::RealArray{S,0.0,Inf}) where {S}
    va = flatvector(adj)
    v∂ = flatvector(∂)
    @avx for i ∈ eachindex(va)
        va[i] = va[i] * v∂[i] + 1.0
    end
end



