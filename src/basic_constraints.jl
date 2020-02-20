
@generated function index_description(::Val{descript}, ::Val{I}) where {descript, I}
    Expr(:call, Expr(:curly, :Val, descript[I]))
end


function stack_pointer_call(::typeof(constrain!), sp::StackPointer, θ::Ptr{Float64}, ::Val{descript}, ::Val{I}) where {descript, I}
    θ = gep(θ, parameter_offset(Val{descript}(), Val{I}()))
    stack_pointer_call(constrain, sp, θ, index_description(Val{descript}(), Val{I}()))
end
function constrain(θ::Ptr{Float64}, ::Val{descript}, ::Val{I}) where {descript, I}
    θ = gep(θ, parameter_offset(Val{descript}(), Val{I}()))
    constrain(θ, index_description(Val{descript}(), Val{I}()))
end

@inline ninvlogit(x) = 1/(1+exp(x))
@inline invlogit(x) = ninvlogit(-x)
constrain(θ::Ptr{Float64}, ::RealScalar{-Inf,Inf}) = load(θ)
constrain(θ::Ptr{Float64}, ::RealScalar{0.0,Inf}) = exp(load(θ))
constrain(θ::Ptr{Float64}, ::RealScalar{Inf,0.0}) = -exp(load(θ))
constrain(θ::Ptr{Float64}, ::RealScalar{0.0,1.0}) = ninvlogit(load(θ))
constrain_pullback(θ::Ptr{Float64}, ::RealScalar{-Inf,Inf}) = (load(θ), ReverseDiffExpressionBase.One())
constrain_pullback(θ::Ptr{Float64}, ::RealScalar{0.0,Inf}) = (x = exp(load(θ)); (x,x))
constrain_pullback(θ::Ptr{Float64}, ::RealScalar{Inf,0.0}) = (x = -exp(load(θ)); (x,x))
constrain_pullback(θ::Ptr{Float64}, ::RealScalar{0.0,1.0}) = (x = ninvlogit(load(θ)); (x, Base.FastMath.mul_fast(x, Base.FastMath.sub_fast(x, 1.0))))


