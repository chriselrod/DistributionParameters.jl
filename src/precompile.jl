function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(DistributionParameters.constrain_lkj_factor_jac_quote),Int64,Int64,Int64,Type{Float64},Symbol,Symbol,Symbol,Symbol})
    precompile(Tuple{typeof(DistributionParameters.constrain_lkj_factor_quote),Int64,Int64,Int64,Type{Float64},Symbol,Symbol})
    precompile(Tuple{typeof(DistributionParameters.corr_cholesky_adjoint_mul_quote),Int64,Type,Bool})
    precompile(Tuple{typeof(DistributionParameters.load_parameter!),Array{Any,1},Array{Any,1},Symbol,Type{CorrCholesky{8,T,L} where L where T},Bool,Module,Symbol,Bool,Bool})
    precompile(Tuple{typeof(DistributionParameters.load_parameter!),Array{Any,1},Array{Any,1},Symbol,Type{RealArray{Tuple{12,8},B,T,2,Tuple{1,12},L,U} where U where L where T where B},Bool,Module,Symbol,Bool,Bool})
end
