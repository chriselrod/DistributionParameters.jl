
struct RealFloat{B,T<:Real,U<:Union{T,Nothing}} <: Real
    r::T
    u::U
end
struct RealArray{S,B,T,N,X,L,U<:Union{Nothing,Ptr{T}}} <: PaddedMatrices.AbstractMutableFixedSizeArray{S,T,N,X,L}
    ptr::Ptr{T}
    utpr::U
end
const RealVector{M,B,T,L,U} = RealArray{Tuple{M},B,T,1,Tuple{1},L,U}
const RealMatrix{M,N,B,T,L,U} = RealArray{Tuple{M,N},B,T,2,Tuple{1,M},L,U}
@inline Base.pointer(A::RealArray) = A.ptr

PaddedMatrices.type_length(::Type{<:RealFloat}) = 1

@inline function PaddedMatrices.LazyMap(f::typeof(Base.log), A::RealArray{S,B,T,N,X,L,Ptr{T}}) where {S,B,T,N,X,L}
    if B === Bounds(zero(T),typemax(T))
        PtrArray{S,T,N,X,L,true}(A.uptr)
    else
        LazyMap{S,T,N,X,L}(SLEEFPirates.log, A.ptr)
    end
end
@inline function PaddedMatrices.LazyMap(f::typeof(SLEEFPirates.log), A::RealArray{S,B,T,N,X,L,Ptr{T}}) where {S,B,T,N,X,L}
    if B === Bounds(zero(T),typemax(T))
        PtrArray{S,T,N,X,L,true}(A.uptr)
    else
        LazyMap{S,T,N,X,L}(SLEEFPirates.log, A.ptr)
    end
end
@inline function PaddedMatrices.LazyMap(f::typeof(SLEEFPirates.logit), A::RealArray{S,B,T,N,X,L,Ptr{T}}) where {S,B,T,N,X,L}
    if B === Bounds(zero(T),one(T))
        PtrArray{S,T,N,X,L,true}(A.uptr)
    else
        LazyMap{S,T,N,X,L}(SLEEFPirates.logit, A.ptr)
    end
end

@inline Base.log(x::RealFloat{B,T,Nothing}) where {B,T} = Base.log(x.r)
@inline function Base.log(x::RealFloat{B,T,T}) where {B,T}
    if B === Bounds(zero(T),typemax(T))
        x.u
    else
        Base.log(x.r)
    end
end
@inline SLEEFPirates.log(x::RealFloat{B,T,Nothing}) where {B,T} = SLEEFPirates.log(x.r)
@inline function SLEEFPirates.log(x::RealFloat{B,T,T}) where {B,T}
    if B === Bounds(zero(T),typemax(T))
        x.u
    else
        SLEEFPirates.log(x.r)
    end
end
@inline function SLEEFPirates.logit(x::RealFloat{B,T,T}) where {B,T}
    if B === Bounds(zero(T),one(T))
        x.u
    else
        SLEEFPirates.logit(x.r)
    end
end


@inline Base.exp(x::RealFloat) = Base.exp(x.r)
@inline SLEEFPirates.exp(x::RealFloat) = SLEEFPirates.exp(x.r)
@inline SpecialFunctions.logabsgamma(x::RealFloat) = SpecialFunctions.logabsgamma(x.r)
@inline SpecialFunctions.logabsbeta(x::RealFloat) = SpecialFunctions.logabsbeta(x.r)
@inline Base.sqrt(x::RealFloat) = Base.FastMath.sqrt_fast(x.r)

@inline Base.:+(x::RealFloat, y) = Base.FastMath.add_fast(x.r, y)
@inline Base.:+(x, y::RealFloat) = Base.FastMath.add_fast(x, y.r)
@inline Base.:+(x::RealFloat, y::RealFloat) = Base.FastMath.add_fast(x.r, y.r)
@inline Base.:-(x::RealFloat, y) = Base.FastMath.sub_fast(x.r, y)
@inline Base.:-(x, y::RealFloat) = Base.FastMath.sub_fast(x, y.r)
@inline Base.:-(x::RealFloat, y::RealFloat) = Base.FastMath.sub_fast(x.r, y.r)
@inline Base.:*(x::RealFloat, y) = Base.FastMath.mul_fast(x.r, y)
@inline Base.:*(x, y::RealFloat) = Base.FastMath.mul_fast(x, y.r)
@inline Base.:*(x::RealFloat, y::RealFloat) = Base.FastMath.mul_fast(x.r, y.r)
@inline Base.:/(x::RealFloat, y) = Base.FastMath.div_fast(x.r, y)
@inline Base.:/(x, y::RealFloat) = Base.FastMath.div_fast(x, y.r)
@inline Base.:/(x::RealFloat, y::RealFloat) = Base.FastMath.div_fast(x.r, y.r)




strip_hashtags(s::String) = s[1+last(findlast("#",s)):end]
strip_hashtags(s::Symbol) = strip_hashtags(string(s))
parameter_names(::Type{<:RealFloat}, s::Symbol) = [strip_hashtags(s)]
@generated function parameter_names(::Type{<: RealArray{S}}, s::Symbol) where {S}
    SV = S.parameters
    N = length(S.parameters)
    L = prod(SV)
    loop = quote
        ind += 1
        names[ind] = ss * "[" * si_1 * "]"
    end
    for n ∈ 1:N
        i_n = Symbol(:i_, n)
        if n == N
            si_ndef = :(string($i_n))
        else
            si_ndef = :(string($i_n) * "," * $(Symbol(:si_, n+1)))
        end
        loop = quote
            for $i_n ∈ 1:$(SV[n])
                $(Symbol(:si_, n)) = $si_ndef
                $loop
            end
        end
    end
    quote
        ss = strip_hashtags(s)
        names = Vector{String}(undef, $L)
        ind = 0
        $loop
        names
    end
end


function load_parameter!(
    first_pass, second_pass, out, ::Type{RealFloat}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, exportparam::Bool = false
)
    T = Float64
    B = Bounds(typemin(T), typemax(T))
    
    load_parameter!(
        first_pass, second_pass, out, RealFloat{B,T}, partial, m, sptr, logjac, exportparam
    )
end


function load_parameter!(
    first_pass, second_pass, out, ::Type{RealFloat{BT}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, exportparam::Bool = false
) where {BT}
    if BT isa Bounds
        B = BT
        T = typeof(BT.lb)
    elseif BT isa DataType
        T = BT
        B = Bounds(typemin(BT), typemax(BT))
    else
        throw("RealFloat parameter $BT not recognized.") 
    end

    load_parameter!(
        first_pass, second_pass, out, RealFloat{B,T}, partial, m, sptr, logjac, exportparam
    )
end


function load_parameter!(
    first_pass, second_pass, out, ::Type{RealFloat{B,T}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, exportparam::Bool = false
) where {T, B}
    shape = Int[]
    load_transformations!(
        first_pass, second_pass, B, out, shape,
        partial, logjac, sptr,
        m, Symbol("##θparameter##"), Symbol("##∂θparameter##"),
        exportparam
    )    
end

function load_parameter!(
    first_pass, second_pass, out, ::Type{<:RealArray{S}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, exportparam::Bool = false
) where {S}
    T = Float64
    B = Bounds(typemin(T), typemax(T))
    
    load_parameter!(
        first_pass, second_pass, out, RealArray{S,B,T}, partial, m, sptr, logjac, exportparam
    )
end


function load_parameter!(
    first_pass, second_pass, out, ::Type{<:RealArray{S,BT}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, exportparam::Bool = false
) where {S,BT}
    if BT isa Bounds
        B = BT
        T = typeof(BT.lb)
    elseif BT isa DataType
        T = BT
        B = Bounds(typemin(BT), typemax(BT))
    else
        throw("RealFloat parameter $BT not recognized.") 
    end

    load_parameter!(
        first_pass, second_pass, out, RealArray{S,B,T}, partial, m, sptr, logjac, exportparam
    )
end

function load_parameter!(
    first_pass, second_pass, out, ::Type{<:RealArray{S,B,T}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, exportparam::Bool = false
) where {S,B,T}
    load_transformations!(
        first_pass, second_pass, B, out, Int[S.parameters...],
        partial, logjac, sptr,
        m, Symbol("##θparameter##"), Symbol("##∂θparameter##"),
        exportparam
    )    
end




