
# Assumes that the ptr does not alias
# Represents this via not actually loading/storing value to the ptr until some finalization step.
struct Reference{T,V}
    ptr::Ptr{T}
    value::V
end

Base.:(+)(r::Reference, x) = Reference(r.ptr, vadd(r.value, x))
Base.:(-)(r::Reference, x) = Reference(r.ptr, vsub(r.value, x))
Base.:(*)(r::Reference, x) = Reference(r.ptr, vmul(r.value, x))
Base.:(/)(r::Reference, x) = Reference(r.ptr, vdiv(r.value, x))
Base.:(+)(x, r::Reference) = Reference(r.ptr, vadd(x, r.value))
Base.:(-)(x, r::Reference) = Reference(r.ptr, vsub(x, r.value))
Base.:(*)(x, r::Reference) = Reference(r.ptr, vmul(x, r.value))
Base.:(/)(x, r::Reference) = Reference(r.ptr, vdiv(x, r.value))
Base.exp(r::Reference) = Reference(r.ptr, exp(r.value))
Base.log(r::Reference) = Reference(r.ptr, log(r.value))

