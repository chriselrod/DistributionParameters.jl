using DistributionParameters
using Test
using ForwardDiff, LinearAlgebra, PaddedMatrices, StructuredMatrices, Random, ProbabilityModels,   
        ProbabilityModels, LoopVectorization, VectorizationBase, SLEEFPirates, SIMDPirates

@testset "DistributionParameters.jl" begin
    # Write your own tests here.

K = 4
S = (@Constant randn(K,4K)) |> x -> x * x'
S *= (1/16)
pS = StructuredMatrices.SymmetricMatrixL(S)
L = StructuredMatrices.lower_chol(S);

T = 8; δₜ = (1/16) * reduce(+, (@Constant randexp(T-1)) for i ∈ 1:8)
times = vcat(zero(ConstantFixedSizePaddedVector{1,Float64}), cumsum(δₜ));
ρ = ConstantFixedSizePaddedVector{4}((0.5,0.55,0.60,0.65));

workspace = (
    Sigfull = PaddedMatrices.PaddedArray{Float64}(undef, (K*T,K*T)),
    ARs = PaddedMatrices.MutableFixedSizePaddedArray{Tuple{T,T,K},Float64}(undef),
    ∂ARs = PaddedMatrices.MutableFixedSizePaddedArray{Tuple{T,T,K},Float64}(undef)
);
Cov, pAR, pL = DistributionParameters.∂CovarianceMatrix(ρ, L, times, workspace, Val((true,true)))

Cov1s = PaddedMatrices.PaddedArray{Float64}(undef, (K*T,K*T)); Cov1s .= 1;





vunstable(x::AbstractVector{T}) where {T} = ConstantFixedSizePaddedArray{Tuple{length(x)},T,1}(ntuple(i -> x[i], length(x)))
x = randn(16,16);
ltx = LowerTriangular(x);
ltxv = vec(ltx);



function trisub2(N)
    out = collect(1:(N+1):N^2)
    ind = 0
    for n ∈ 0:N-1
        ind += n+1
        r = N-n-1
        for i ∈ 1:r
            ind += 1
            push!(out, ind)
        end
    end
    out
end
ltxv2 = ltx[trisub2(16)];
trivec = vunstable(ltxv2);

r2 = @Constant randn(120);
@generated function load_lkj(a::PaddedMatrices.AbstractFixedSizePaddedVector{L,T}) where {L,T}
    fp = quote end
    sp = quote end
    out = gensym(:L)
    M = (Int(sqrt(1 + 8L))-1)>>1
    DistributionParameters.load_parameter(fp.args, sp.args, out, DistributionParameters.LKJ_Correlation_Cholesky{M+1}, false)
    quote
        $(Symbol("##θparameter##")) = VectorizationBase.vectorizable(a);
        target = zero($T)
        $fp
        $sp
        $out, target
    end
end
@generated function ∂load_lkj(a::PaddedMatrices.AbstractFixedSizePaddedVector{L,T}, seedout = (M = (Int(sqrt(1 + 8L))-1)>>1; fill(1.0, PaddedMatrices.Static{(((M+2)*(M+1))>>1)}()))) where {L,T}
    fp = quote end
    sp = quote end
    out = gensym(:L)
    M = (Int(sqrt(1 + 8L))-1)>>1
    DistributionParameters.load_parameter(fp.args, sp.args, out, DistributionParameters.LKJ_Correlation_Cholesky{M+1}, true)
    quote
        $(Symbol("##∂θparameter##m")) = PaddedMatrices.MutableFixedSizePaddedVector{L,T}(undef);
        $(Symbol("##∂θparameter##")) = VectorizationBase.vectorizable($(Symbol("##∂θparameter##m")))
        $(Symbol("##θparameter##")) = VectorizationBase.vectorizable(a);
        target = zero($T)
        $fp
        $(Symbol("###seed###", out)) = seedout'
        $sp
        $out, target, $(Symbol("##∂θparameter##m"))
    end
end
Lm, t, partial = ∂load_lkj(r2, trivec); Lm

r2v = Array(r2);
nr2vl = @. 1 / (1 + exp(r2v));
r2vl = 1 .- nr2vl;
zvals = @. 1 - 2 * nr2vl;
r2vlpartial = @. r2vl * nr2vl;
r2l = vunstable(zvals);

lkj, lkjac, lkjldgrad, lkjgrad = DistributionParameters.∂lkj_constrain(r2l);
j = ForwardDiff.jacobian(x -> Array(DistributionParameters.lkj_constrain(vunstable(x))[1]), r2l);
ltxvj = ltxv' * j;
@test all(ltxvj .≈ trivec' * lkjgrad) # matches
g = ForwardDiff.gradient(x -> DistributionParameters.lkj_constrain(vunstable(x))[2], r2l); g'
@test all(lkjldgrad .≈ g)


end
