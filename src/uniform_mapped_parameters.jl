
struct RealFloat{B,T} <: Real end
struct RealArray{S,B,T,N,P,L} <: PaddedMatrices.AbstractMutableFixedSizePaddedArray{S,T,N,P,L} end
const RealVector{M,B,T,L} = RealArray{Tuple{M},B,T,1,L,L}
const RealMatrix{M,N,B,T,L} = RealArray{Tuple{M,N},B,T,2,M,L}

PaddedMatrices.type_length(::Type{<:RealFloat}) = 1


strip_hashtags(s::String) = s[1+last(findlast("#",s)):end]
strip_hashtags(s::Symbol) = strip_hashtags(string(s))
parameter_names(::Type{<:RealFloat}, s::Symbol) = [strip_hashtags(s)]
@generated function parameter_names(::Type{<: RealArray{S}}, s::Symbol) where {S}
    SV = S.parameters
    N = length(S.parameters)
    L = prod(SV)
#    indices = [Symbol(:i_, n) for n ∈ 1:N]
    loop = quote
#        for i_1 ∈ 1:$(SV[1])
            ind += 1
            names[ind] = ss * "[" * si_1 * "]"
#        end
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




