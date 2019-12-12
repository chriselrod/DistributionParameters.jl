function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(DistributionParameters.constrain_lkj_factor_jac_quote),Int64,Int64,Int64,Type{Float64},Symbol,Symbol,Symbol,Symbol})
    precompile(Tuple{typeof(DistributionParameters.constrain_lkj_factor_quote),Int64,Int64,Int64,Type{Float64},Symbol,Symbol})
    precompile(Tuple{typeof(DistributionParameters.corr_cholesky_adjoint_mul_quote),Int64,Type,Bool})
    precompile(Tuple{typeof(DistributionParameters.load_parameter!),Array{Any,1},Array{Any,1},Symbol,Type{CorrCholesky{8,T,L} where L where T},Bool,Module,Symbol,Bool,Bool})
    precompile(Tuple{typeof(DistributionParameters.load_parameter!),Array{Any,1},Array{Any,1},Symbol,Type{RealArray{Tuple{12,8},B,T,2,Tuple{1,12},L,U} where U where L where T where B},Bool,Module,Symbol,Bool,Bool})
    precompile(Tuple{typeof(DistributionParameters.load_parameter!),Array{Any,1},Array{Any,1},Symbol,Type{RealArray{Tuple{8},B,T,1,Tuple{1},L,U} where U where L where T where B},Bool,Module,Symbol,Bool,Bool})
    precompile(Tuple{typeof(DistributionParameters.load_parameter!),Array{Any,1},Array{Any,1},Symbol,Type{RealArray{Tuple{8},Bounds{Float64}(0.0, Inf),T,1,Tuple{1},L,U} where U where L where T},Bool,Module,Symbol,Bool,Bool})
    precompile(Tuple{typeof(PaddedMatrices.type_length),Type{CorrCholesky{8,T,L} where L where T}})
    precompile(Tuple{typeof(hash),Pair{Int64,Bounds{Float64}},UInt64})
end
