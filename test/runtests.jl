using DistributionParameters
using Test


using StaticArrays, SLEEF, BenchmarkTools, LoopVectorization

@inline function my_dot(x, y)
    out = zero(promote_type(eltype(x),eltype(y)))
    @inbounds @fastmath for i ∈ eachindex(x, y)
        out += x[i] * y[i]
    end
    out
end
function my_exp_dot_test(x, y)
    w = MArray(x)
    z = MArray(y)
    @inbounds for i ∈ eachindex(w, z)
        w[i] = SLEEF.exp(w[i])
        z[i] = SLEEF.exp(z[i])
    end
    # x = SArray(w)
    # y = SArray(z)
    my_dot(w, z)
    # x' * y
end

function my_exp_dot_test2(x, y)
    w = MArray(x)
    z = MArray(y)
    @vectorize for i ∈ eachindex(w)
        w[i] = exp(w[i])
        z[i] = exp(z[i])
    end
    x = SArray(w)
    y = SArray(z)
    my_dot(x, y)
end
function my_exp_dot_test3(a::MVector{32,T}) where {T}
    w = MVector{16,T}(undef)
    @inbounds for i ∈ 1:16
        w[i] = a[i]
    end
    z = MVector{16,T}(undef)
    @inbounds for i ∈ 1:16
        z[i] = a[i+16]
    end
    @inbounds for i ∈ eachindex(w, z)
        w[i] = SLEEF.exp(w[i])
        z[i] = SLEEF.exp(z[i])
    end
    x = SArray(w)
    y = SArray(z)
    my_dot(x, y)
end

function fastmul(A::PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{M,N,T},
                B::PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{N,P,T}) where {M,N,P,T}
    W = MutableFixedSizePaddedMatrix(A)
    X = MutableFixedSizePaddedMatrix(B)
    C = MutableFixedSizePaddedMatrix{M,P,T}(undef)
    mul!(C, W, X)
    ConstantFixedSizePaddedMatrix(C)
end

@testset "DistributionParameters.jl" begin
    # Write your own tests here.


    inv_logit(x) = 1 / (1 + exp(-x))
    ∂inv_logit(x) = (ilx = inv_logit(x); ilx * (1-ilx))
    log∂inv_logit(x) = log(∂inv_logit(x))

end
