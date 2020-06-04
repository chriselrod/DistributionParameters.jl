
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
# @generated function index_description(::Val{descript}, ::Val{I}) where {descript, I}
#     Expr(:call, Expr(:curly, :Val, descript[I]))
# end
# function stack_pointer_call(::typeof(constrain), sp::StackPointer, θ::Ptr{Float64}, ::Val{descript}, ::Val{I}) where {descript, I}
#     θ = gep(θ, parameter_offset(Val{descript}(), Val{I}()))
#     stack_pointer_call(constrain, sp, θ, index_description(Val{descript}(), Val{I}()))
# end
# function constrain(θ::Ptr{Float64}, ::Val{descript}, ::Val{I}) where {descript, I}
#     θ = gep(θ, parameter_offset(Val{descript}(), Val{I}()))
#     constrain(θ, index_description(Val{descript}(), Val{I}()))
# end

@inline constrain(θ::Ptr{Float64}, i, ::RealScalar{-Inf,Inf}) = (Zero(), vload(gep(θ, i)))
@inline constrain(θ::Ptr{Float64}, i, ::RealScalar{0.0,Inf}) = (x = vload(gep(θ, i)); (x, exp(x)))
@inline constrain(θ::Ptr{Float64}, i, ::RealScalar{Inf,0.0}) = (x = vload(gep(θ, i)); (x, Base.FastMath.sub_fast(exp(x))))
@inline constrain(θ::Ptr{Float64}, i, ::RealScalar{L,Inf}) where {L} = (x = vload(gep(θ, i)); (x, Base.FastMath.add_fast(L, exp(x))))
@inline constrain(θ::Ptr{Float64}, i, ::RealScalar{Inf,U}) where {U} = (x = vload(gep(θ, i)); (x, Base.FastMath.sub_fast(U, exp(x))))
# @inline constrain(θ::Ptr{Float64}, ::RealScalar{L,U}) where {L,U} = (x = ninvlogit(vload(gep(θ, i))); (muladd(Base.FastMath.sub_fast(x),x,x), x))
@inline constrain_pullback(θ::Ptr{Float64}, i, ::RealScalar{-Inf,Inf}) = (Zero(), vload(gep(θ, i)), ReverseDiffExpressionBase.One())
@inline function constrain_pullback(θ::Ptr{Float64}, i, ::RealScalar{0.0,Inf})
    y = vload(gep(θ, i))
    x = exp(y)
    (y, x, x)
end
@inline function constrain_pullback(θ::Ptr{Float64}, i, ::RealScalar{-Inf,0.0})
    y = vload(gep(θ, i))
    x = Base.FastMath.sub_fast(exp(y))
    (y, x, x)
end
@inline function constrain_pullback(θ::Ptr{Float64}, i, ::RealScalar{L,Inf}) where {L}
    y = vload(gep(θ, i))
    x = exp(y)
    (y, Base.FastMath.add_fast(L, x), x)
end
@inline function constrain_pullback(θ::Ptr{Float64}, i, ::RealScalar{-Inf,U}) where {U}
    y = vload(gep(θ, i))
    x = Base.FastMath.sub_fast(exp(y))
    (y, Base.FastMath.add_fast(U, x), x)
end
# constrain_pullback(θ::Ptr{Float64}, ::RealScalar{0.0,1.0}) = (x = ninvlogit(vload(gep(θ, i))); (x, Base.FastMath.mul_fast(x, Base.FastMath.sub_fast(x, 1.0))))
@inline alloc_adj(::Any, ::Any, ::RealScalar) = Zero()
constrain_reverse!(∇::Ptr{T}, adj::T, ∂::One, ::RealScalar) where {T} = store!(∇, adj)
constrain_reverse!(∇::Ptr, adj, ∂::One, ::RealScalar) = store!(∇, vsum(adj))
constrain_reverse!(∇::Ptr{T}, adj::T, ∂, ::RealScalar) where {T} = store!(∇, vmul(adj, ∂))
constrain_reverse!(∇::Ptr, adj, ∂, ::RealScalar) = store!(∇, vsum(vmul(adj, ∂)))

@inline alloc_adj(∇::Ptr{Float64}, ::RealArray{S,<:Any,<:Any,0}) where {S} = NoPadPtrView{S}(∇)
@inline constrain(θ::Ptr{Float64}, ::RealArray{S,-Inf,Inf,0}) where {S} = Zero(), NoPadPtrView{S}(θ)
@inline constrain_pullback(θ::Ptr{Float64}, ::RealArray{S,-Inf,Inf,0}) where {S} = (Zero(), NoPadPtrView{S}(θ), One())
@inline stack_pointer_call(::typeof(constrain), sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,-Inf,Inf,0}) where {S} = (sp, Zero(), NoPadPtrView{S}(θ))
@inline stack_pointer_call(::typeof(constrain_pullback), sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,-Inf,Inf,0}) where {S} = (sp, Zero(), NoPadPtrView{S}(θ), One())
# @inline alloc_adj(∇::Ptr{Float64}, ::RealArray{S,-Inf,Inf,0}) where {S} = NoPadPtrView{S}(∇)
@inline constrain_reverse!(adj::PtrArray{S}, ∂::One, ::RealArray{S,-Inf,Inf}) where {S} = nothing

# @inline constrain(θ::Ptr{Float64}, ::RealArray{S,0.0,Inf,0}) where {S} = NoPadPtrView{S}(θ)
# @inline constrain_pullback(θ::Ptr{Float64}, ::RealArray{S,0.0,Inf,0}) where {S} = (NoPadPtrView{S}(θ), ReverseDiffExpressionBase.One())
# @inline alloc_adj(∇::Ptr{Float64}, ::RealArray{S,0.0,Inf}) where {S} = NoPadPtrView{S}(∇)
# @inline constrain_reverse!(∇::Ptr{Float64}, adj::PtrArray{S}, ∂::One, ::RealArray{S,0.0,Inf}) = nothing

@inline function constrain_bounded_array!(ev::AbstractArray, uv::AbstractArray, ::RealArray{S,0.0,Inf,0}) where {S}
    t = vzero()
    @avx for i ∈ eachindex(uv)
        euv = exp(uv[i])
        t += euv
        ev[i] = euv
    end
    t
end
@inline function constrain_bounded_array!(ev::AbstractArray, uv::AbstractArray, ::RealArray{S,L,Inf,0}) where {L,S}
    t = vzero()
    @avx for i ∈ eachindex(uv)
        euv = exp(uv[i])
        t += euv
        ev[i] = L + euv
    end
    t
end
@inline function constrain_bounded_array!(ev::AbstractArray, uv::AbstractArray, ::RealArray{S,-Inf,0.0,0}) where {S}
    t = vzero()
    @avx for i ∈ eachindex(uv)
        euv = exp(uv[i])
        t += euv
        ev[i] = - euv
    end
    t
end
@inline function constrain_bounded_array!(ev::AbstractArray, uv::AbstractArray, ::RealArray{S,-Inf,U,0}) where {U,S}
    t = vzero()
    @avx for i ∈ eachindex(uv)
        euv = exp(uv[i])
        t += euv
        ev[i] = U - euv
    end
    t
end
@inline function constrain_single_bound_spc(sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,L,U,0}) where {L,U,S}
    sp, e = NoPadPtrView{S}(sp)
    uv = flatvector(NoPadPtrView{S}(θ))
    t = constrain_bounded_array!(flatvector(e), uv, RealArray{S,L,U,0}())
    sp, (t, MappedArray(e, pointer(uv)))    
end
@inline function constrain_pullback_single_bound_spc(sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,L,U,0}) where {L,U,S}
    sp, e = NoPadPtrView{S}(sp)
    uv = flatvector(NoPadPtrView{S}(θ))
    t = constrain_bounded_array!(flatvector(e), uv, RealArray{S,L,U,0}())
    sp, (t, MappedArray(e, pointer(uv)), e)
end
@inline function stack_pointer_call(::typeof(constrain), sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,L,Inf,0}) where {L,S}
    constrain_single_bound_spc(sp, θ, RealArray{S,L,Inf,0}())
end
@inline function stack_pointer_call(::typeof(constrain), sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,-Inf,U,0}) where {U,S}
    constrain_single_bound_spc(sp, θ, RealArray{S,-Inf,U,0}())
end
@inline function stack_pointer_call(::typeof(constrain_pullback), sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,L,Inf,0}) where {L,S}
    constrain_pullback_single_bound_spc(sp, θ, RealArray{S,L,Inf,0}())
end
@inline function stack_pointer_call(::typeof(constrain_pullback), sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,-Inf,U,0}) where {U,S}
    constrain_pullback_single_bound_spc(sp, θ, RealArray{S,-Inf,U,0}())
end

# @inline alloc_adj(∇::Ptr{Float64}, ::RealArray{S,-Inf,Inf,0}) where {S} = NoPadPtrView{S}(∇)
# function stack_pointer_call(::typeof(alloc_adj), sp::StackPointer, θ::Ptr{Float64}
function constrain_reverse!(adj::PtrArray{S}, ∂::PtrArray{S}, ::RealArray{S,0.0,Inf}) where {S}
    @avx for i ∈ eachindex(va)
        adj[i] = adj[i] * ∂[i] + 1.0
    end
end
function constrain_reverse!(adj::PtrArray{S}, ∂::PtrArray{S}, ::RealArray{S,L,Inf}) where {L,S}
    @avx for i ∈ eachindex(va)
        adj[i] = adj[i] * (∂[i]-L) + 1.0
    end
end


# @inline function constrain_bounded_array!(ev::AbstractArray, uv::AbstractArray, ::RealArray{S,0.0,Inf,0}) where {S}
#     t = vzero()
#     @avx for i ∈ eachindex(uv)
#         euv = exp(uv[i])
#         t += euv
#         ev[i] = euv
#     end
#     t, uv
# end
# @inline function constrain_lower_bounded_array!(ev::AbstractArray, uv::AbstractArray, ::RealArray{S,L,Inf,0}) where {L,S}
#     t = vzero()
#     @avx for i ∈ eachindex(uv)
#         euv = exp(uv[i])
#         t += euv
#         ev[i] = L + euv
#     end
#     sp, t, e, uv
# end
# @inline alloc_adj(∇::Ptr{Float64}, ::RealArray{S,-Inf,Inf,0}) where {S} = NoPadPtrView{S}(∇)
# function stack_pointer_call(::typeof(alloc_adj), sp::StackPointer, θ::Ptr{Float64}
# function constrain_reverse!(adj::PtrArray{S}, ∂::PtrArray{S}, ::RealArray{S,0.0,Inf}) where {S}
#     @avx for i ∈ eachindex(va)
#         adj[i] = adj[i] * ∂[i] + 1.0
#     end
# end



