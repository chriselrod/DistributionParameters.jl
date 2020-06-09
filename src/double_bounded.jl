

struct InvLogitElement{T<:Real} <: Real
    data::T
    logp::T
    nlogitp::T
end
struct InvLogitVec{W,T<:Real} <: AbstractStructVec{W,T}
    data::Vec{W,T}
    logp::Vec{W,T}
    nlogitp::Vec{W,T}
end
const InvLogitValue{T} = Union{InvLogitElement{T},InvLogitVec{<:Any,T}}
@inline invlogitwrap(p::T, logp::T, invlogit::T) where {T<:Number} = InvLogitElement(p, logp, invlogit)
@inline invlogitwrap(p::Vec{W,T}, logp::Vec{W,T}, invlogit::Vec{W,T}) where {W,T<:Number} = InvLogitVec(p, logp, invlogit)
@inline function invlogitwrap(p::AbstractStructVec{W,T}, logp::AbstractStructVec{W,T}, invlogit::AbstractStructVec{W,T}) where {W,T<:Number}
    InvLogitVec(extract_data(p), extract_data(logp), extract_data(invlogit))
end
struct InvLogitStridedPointer{L,T,P<:AbstractStridedPointer{T}} <: AbstractStridedPointer{T}
    ptr::P
    logitptr::Ptr{T}
    @inline InvLogitStridedPointer{L}(ptr::P, logitptr::Ptr{T}) where {L,T,P<:AbstractStridedPointer} = InvLogitStridedPointer{L,T,P}(ptr, logitptr)
end
@inline InvLogitStridedPointer(ptr::P, logitptr::Ptr{T}, ::Val{L}) where {L,T,P<:AbstractStridedPointer} = InvLogitStridedPointer{L,T,P}(ptr, logitptr)
@inline Base.convert(::Type{T1}, a::InvLogitElement{T2}) where {T1,T2} = convert(T1, a.data)
@inline Base.promote_type(::Type{InvLogitElement{T1}}, ::Type{T2}) where {T1,T2} = promote_type(T1, T2)
@inline Base.promote_type(::Type{T2}, ::Type{InvLogitElement{T1}}) where {T1,T2} = promote_type(T1, T2)
@inline Base.promote_type(::Type{InvLogitElement{T1}}, ::Type{InvLogitElement{T2}}) where {T1,T2} = promote_type(T1, T2)
@inline Base.pointer(A::InvLogitStridedPointer) = A.ptr.ptr
@inline VectorizationBase.offset(p::InvLogitStridedPointer, i::Tuple) = offset(p.ptr, i)
@inline function VectorizationBase.vload(ilp::InvLogitStridedPointer{L,T}, i) where {L,T}
    o = offset(ilp, i)
    p = vload(ilp.ptr, o)
    logp = vload(ilp.ptr, o + align(L,T))
    logitp = vload(ilp.logitptr, o)
    invlogitwrap(p, logp, logitp)
end
@inline Base.log(il::InvLogitValue) = il.logp
@inline SLEEFPirates.log1m(il::InvLogitValue) = vadd(il.logp, il.nlogitp)
@inline SLEEFPirates.logit(il::InvLogitValue) = vsub(il.nlogitp)
@inline SLEEFPirates.nlogit(il::InvLogitValue) = il.nlogitp


struct InvLogitArray{S,T,N,X,SN,XN} <: AbstractStrideArray{S,T,N,X,SN,XN,true}
    ptr::Ptr{T}
    logitptr::Ptr{T}
    size::NTuple{SN,UInt32}
    stride::NTuple{XN,UInt32}
end
@generated function InvLogitArray{S}(ptr::Ptr{T}, logitptr::Ptr{T}) where {S,T}
    N, X, L = calc_NXL(S.parameters, T, (S.parameters[1])::Int)
    L = VectorizationBase.align(L, T)
    Expr(:block, Expr(:meta,:inline), :(InvLogitArray{$S,$T,$N,$X,0,0}(ptr, logitptr, tuple(), tuple())))
end
@inline Base.pointer(A::InvLogitArray) = A.ptr
@inline function VectorizationBase.stridedpointer(A::InvLogitArray{S,T,N,X,SN,XN}) where {S,T,N,X,SN,XN}
    InvLogitStridedPointer(stridedpointer(PtrArray(A)), A.logitptr, memory_length_val(A))
end

@inline function constrain(θ::Ptr{Float64}, ::RealScalar{0.0,1.0})
    y = vload(θ)
    p = ninvlogit(y)
    logp = log(p)
    t = Base.FastMath.add_fast(Base.FastMath.mul_fast(2, logp), y)
    t, InvLogitElement(p, logp, y)
end
@inline function constrain_pullback(∇::Ptr{Float64}, θ::Ptr{Float64}, ::RealScalar{0.0,1.0})
    y = vload(θ)
    p = ninvlogit(y)
    logp = log(p)
    t = Base.FastMath.add_fast(Base.FastMath.mul_fast(2, logp), y)
    ile = InvLogitElement(p, logp, y)
    (t, ile), (Zero(), ile)
end
@inline function constrain_double_bounded_array!(c::AbstractStrideMatrix{N,2}, nil::AbstractVector, ::RealArray{S,0.0,1.0,0}) where {N,S}
    t = vzero()
    @avx for i ∈ eachindex(nil)
        y = nil[i]
        p = 1 / (1 + exp(y))
        logp = log(p)
        t += 2*logp + y
        c[i,1] = p
        c[i,2] = logp
    end
    t
end
@inline function stack_pointer_call(::typeof(constrain), sp::StackPointer, θ::Ptr{Float64}, ::RealArray{S,0.0,1.0,0}) where {S}
    sp, e = NoPadFlatPtrViewMulti(sp, S, Val{2}())
    uv = flatvector(NoPadPtrView{S}(θ))
    t = constrain_double_bounded_array!(flatvector(e), uv, RealArray{S,L,U,0}())
    ila = InvLogitArray{S}(pointer(e), pointer(θ))
    sp, (t, ila)
end
@inline function stack_pointer_call(::typeof(constrain_pullback!), sp::StackPointer, ∇::Ptr{Float64}, θ::Ptr{Float64}, ::RealArray{S,0.0,1.0,0}) where {S}
    sp, e = NoPadFlatPtrViewMulti(sp, S, Val{2}())
    uv = flatvector(NoPadPtrView{S}(θ))
    t = constrain_double_bounded_array!(flatvector(e), uv, RealArray{S,L,U,0}())
    ila = InvLogitArray{S}(pointer(e), pointer(θ))
    sp, ((t, ila), (NoPadPtrView{S}(∇), ila))
end


