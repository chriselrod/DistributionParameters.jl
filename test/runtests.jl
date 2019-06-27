using DistributionParameters
using Test
using ForwardDiff, LinearAlgebra, PaddedMatrices, StructuredMatrices, Random, ProbabilityModels,   
        ProbabilityModels, LoopVectorization, VectorizationBase, SLEEFPirates, SIMDPirates

@testset "DistributionParameters.jl" begin
    # Write your own tests here.

K = 7
S = (@Constant randn(K,4K)) |> x -> x * x'
S *= (1/32)
pS = StructuredMatrices.SymmetricMatrixL(S)
L = StructuredMatrices.lower_chol(S);
    L * L'
T = 36; δₜ = (1/16) * reduce(+, (@Constant randexp(T-1)) for i ∈ 1:8);
    times = vcat(zero(ConstantFixedSizePaddedVector{1,Float64}), cumsum(δₜ));
    ρtup = (0.35,0.4,0.45,0.5,0.55,0.60,0.65);
    ρ = ConstantFixedSizePaddedVector{7}(ρtup);
    
KT = K*T;

    ARs = PaddedMatrices.MutableFixedSizePaddedArray{Tuple{T,T,K},Float64}(undef);

    using PaddedMatrices: DynamicPaddedArray
    
workspace = (
    Sigfull = PaddedMatrices.DynamicPaddedArray{Float64}(undef, (K*T,K*T), (KT+15) & -8),
    ARs = ARs,
    ∂ARs = PaddedMatrices.MutableFixedSizePaddedArray{Tuple{T,T,K},Float64}(undef)
);
fill!(workspace.Sigfull, 19.5);
Cov, pAR, pL = DistributionParameters.∂DynamicCovarianceMatrix(ρ, L, times, workspace, Val((true,true)));
    using BenchmarkTools
#pL |> typeof
#spL |> typeof
#spL    
    mtimes = MutableFixedSizePaddedVector(times);
    sp, (spC, spAR, spL) = DistributionParameters.∂CovarianceMatrix(ProbabilityModels.STACK_POINTER, ρ, L, mtimes, Val((true,true)));
#    sp0 = PaddedMatrices.StackPointer(Libc.malloc(8*1<<28));

#    sp, (spC, spAR, spL) = DistributionParameters.∂DynamicCovarianceMatrix(sp0, ρ, L, mtimes, Val((true,true)));
    @benchmark DistributionParameters.∂DynamicCovarianceMatrix($ρ, $L, $times, $workspace, Val((true,true)))
    @benchmark DistributionParameters.∂CovarianceMatrix(ProbabilityModels.STACK_POINTER, $ρ, $L, $mtimes, Val((true,true)))
#exit()
    @code_warntype DistributionParameters.∂DynamicCovarianceMatrix(sp0, ρ, L, mtimes, Val((true,true)),Val(true));
#typeof(L)
spL.ARs
pL.ARs

@benchmark PaddedMatrices.PtrArray{Tuple{36,36,7}}($sp)
@benchmark PaddedMatrices.PtrArray{Tuple{36,36,7},Float64,3,40}($sp)
@benchmark PaddedMatrices.PtrArray{Tuple{36,36,7},Float64,3,40}(Base.unsafe_convert(Ptr{Float64},$sp.ptr))
    @benchmark pointer($sp, Float64) + 121
@code_warntype pointer(sp,Float64)

@code_warntype PaddedMatrices.PtrArray{Tuple{36,36,7},Float64,3,40}(sp)
@code_warntype PaddedMatrices.PtrArray{Tuple{36,36,7},Float64,3,40}(Base.unsafe_convert(Ptr{Float64},sp.ptr))

    
    
@code_warntype PaddedMatrices.PtrArray{Tuple{36,36,7},Float64,3,40}(sp0)
    
    aspl = Array(spL.ARs);
    apl = Array(pL.ARs);
    @test all(aspl .≈ apl)
    asl = Array(spL.LKJ);
    al = Array(pL.LKJ);
    @test all(asl .≈ al)
    asC = Array(Symmetric(spC,:L));
    aC = Array(Symmetric(Cov,:L));
    @test all(asC .≈ aC)
    L
    L_K
    Lm = MutableLowerTriangularMatrix(Array(L))
    ρ, ρs
    Cov, pAR, pL = DistributionParameters.∂DynamicCovarianceMatrix(ρs, L_K, time, workspace, Val((true,true)));
    sp, (spC, spAR, spL) = DistributionParameters.∂CovarianceMatrix(ProbabilityModels.STACK_POINTER, ρ, Lm, mtimes, Val((true,true)));
    asC = Array(Symmetric(spC,:L));
    aC = Array(Symmetric(Cov,:L));
    findall(Symmetric(Array(Cov),:L) .≉  Symmetric(Array(spC),:L) )
    cholesky(Symmetric(aC))

    times, time
    
    size(spC)    
    Symmetric(spC, :L)
    Symmetric(Cov, :L)
spAR
    size(spC)
    size(Cov)
    function fill_block_diag(B::AbstractArray{T,3}) where {T}
        M, N, K = size(B)
        A = zeros(M*K,N*K)
        for k ∈ 0:K-1
            A[1+k*M:(k+1)*M,1+k*N:(k+1)*N] .= @view B[:,:,k+1]
        end
        A
    end
    Lfull = kron(L, Matrix{Float64}(I, T, T)); ARfull = fill_block_diag(pL.ARs);

    @test ( Lfull * ARfull * Lfull' .≈ aC ) |> all
    @test ( Lfull * ARfull * Lfull' .≈ asC ) |> all
    cholesky(Symmetric(aC))

    
    Cov = DistributionParameters.CovarianceMatrix(ρ, L, times, workspace); Cov
#    Cov1s = PaddedMatrices.PaddedArray{Float64}(undef, (K*T,K*T)); Cov1s .= 1;
using Statistics

    # some disagreements!
    #
    
    
    using ForwardDiff, StaticArrays


     function DistributionParameters.CovarianceMatrix(
         rhos::PaddedMatrices.AbstractFixedSizePaddedVector{K,T1},
         L::StructuredMatrices.AbstractLowerTriangularMatrix{K,T2},
         times::ConstantFixedSizePaddedVector{nT}
     ) where {K,T1,T2,nT}

         T = promote_type(T1, T2)
         workspace = (
             Sigfull = PaddedMatrices.DynamicPaddedArray{T}(undef, (K*nT,K*nT)),
             ARs = PaddedMatrices.MutableFixedSizePaddedArray{Tuple{nT,nT,K},T}(undef),
         )
         DistributionParameters.DynamicCovarianceMatrix(rhos, L, times, workspace)
         
     end

    ForwardDiff.gradient(r -> sum(Symmetric(DistributionParameters.CovarianceMatrix( ConstantFixedSizePaddedVector{7}(ntuple(i -> r[i], Val(7))), L, times))), SVector(ρtup))

zd = ones(K*T,K*T) ; #.- LowerTriangular(ones(K*T,K*T));

    @test all(
        ForwardDiff.gradient(r -> sum(Symmetric(zd .* DistributionParameters.CovarianceMatrix( ConstantFixedSizePaddedVector{7}(ntuple(i -> r[i], Val(7))), L, times),:L)), SVector(ρtup))'
        .≈
        Array(zd * pAR)
    )
    zr = randn(KT,KT); @. zr = zr + zr';
    @test all(
        ForwardDiff.gradient(r -> sum(Symmetric(zr .* DistributionParameters.CovarianceMatrix( ConstantFixedSizePaddedVector{7}(ntuple(i -> r[i], Val(7))), L, times),:L)), SVector(ρtup))'
        .≈
        Array(zr * pAR)
    )

    @generated function StructuredMatrices.LowerTriangularMatrix(A::SMatrix{K,K,T}) where {K,T}
        outtup = Expr(:tuple, )
        for k ∈ 1:K
            push!(outtup.args, :(A[$k,$k]))
        end
        for kc ∈ 1:K, kr ∈ kc+1:K
            push!(outtup.args, :(A[$kr,$kc]))
        end
        :(@inbounds LowerTriangularMatrix{$K,$T,$(length(outtup.args))}( $outtup ) )
    end

    sL = SMatrix{K,K}(Array(L))
    LowerTriangularMatrix(sL)
    L
    L.data
    ForwardDiff.gradient(r -> sum(Symmetric(DistributionParameters.CovarianceMatrix( ρ, LowerTriangularMatrix(r), times ))), sL )
    ForwardDiff.gradient(r -> sum(UpperTriangular(DistributionParameters.CovarianceMatrix( ρ, LowerTriangularMatrix(r), times ))), sL )
    @test all(
        LowerTriangular(ForwardDiff.gradient(r -> (Lf = kron(r, Matrix{Float64}(I,T,T)); sum(Lf * ARfull * Lf')), sL))
        .≈
        Array( zd * pL )
    )


    ForwardDiff.gradient(r -> sum( Symmetric(zr .* DistributionParameters.CovarianceMatrix( ρ, LowerTriangularMatrix(r), times ))), sL )
    @test all(
        LowerTriangular(ForwardDiff.gradient(r -> (Lf = kron(r, Matrix{Float64}(I,T,T)); sum(zr .* (Lf * ARfull * Lf'))), sL))
        .≈
        Array( zr * pL )
    )
    
    sum(ARs, dims = (1,2))
    

    Ctest = zeros(KT,KT); Ctest[1+(K-1)*T:KT, 1+(K-1)*T:KT] .= pAR.∂ARs[:,:,K];
    (Lfull * Ctest * Lfull') |> sum

    ForwardDiff.gradient(
        r -> sum(UpperTriangular(zd .* DistributionParameters.CovarianceMatrix( ConstantFixedSizePaddedVector{4}(ntuple(i -> r[i], Val(4))), L, times))), SVector(0.5,0.65,0.7,0.75))
    

    
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
