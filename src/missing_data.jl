

"""
Parameters are:
M: number Missing
B: Bounds
T: DataType
N: Dimensionality of Array (how many axis?)
"""
struct MissingDataArray{M,B,T,N,A <: AbstractArray{T,N}}
    data::A
    inds::Vector{Int}
#    inds::MutableFixedSizePaddedVector{M,Int,M,M}
end

function parameter_names(::Type{<:MissingDataArray{M}}, s::Symbol) where {M}
    ss = strip_hashtags(s) * "_missing_#"
    [ss * string(m) for m ∈ 1:M]
end

PaddedMatrices.type_length(::MissingDataArray{M}) where {M} = M
PaddedMatrices.type_length(::Type{<:MissingDataArray{M}}) where {M} = M
PaddedMatrices.param_type_length(::MissingDataArray{M}) where {M} = M
PaddedMatrices.param_type_length(::Type{<:MissingDataArray{M}}) where {M} = M

"""
This function is not type stable.
"""
function maybe_missing(A::AA) where {T,N,AA <: AbstractArray{Union{Missing,T},N}}
    M = sum(ismissing, A)
    M == 0 && return convert(Array{T}, A)
    l, u = extrema(skipmissing(A))
    lb = l > 0 ? zero(T) : -typemax(T)
    ub = u < 0 ? zero(T) :  typemax(T)
    convert(MissingDataArray{M,Bounds(lb,ub)}, A)
end
maybe_missing(A::AbstractArray) = A

function Base.convert(::Type{<:MissingDataArray{M}}, A::AbstractArray{Union{Missing,T}}) where {M,T}
    convert(MissingDataArray{M,Bounds(typemin(T),typemax(T))}, A)
end
function Base.convert(::Type{<:MissingDataArray{M,B}}, A::AA) where {M,B,T,N,AA <: AbstractArray{Union{Missing,T},N}}
#    M = sum(ismissing, A)
    #    M == 0 &&
    data = similar(A, Float64)
    ptr_A = Base.unsafe_convert(Ptr{T}, pointer(A)) # is this necessary?
    LoopVectorization.@vvectorize for i ∈ eachindex(A)
        data[i] = ptr_A[i]
    end
    inds = findall(ismissing, vec(A))
    MissingDataArray{M,B,T,N,typeof(data)}(
        data, inds
    )
end


function load_missing_as_vector!(
    first_pass, second_pass, out, ::Type{<:MissingDataArray{M,B,T}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, copyexport::Bool = false
) where {M,B,T}
    load_parameter!(
        first_pass, second_pass, out, RealVector{M,B,T}, partial, m, sptr, logjac, copyexport
    )
end


function load_parameter!(
    first_pass, second_pass, out, ::Type{<:MissingDataArray{M,B,T}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, copyexport::Bool = false
) where {M,B,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    out_missing = Symbol("##missing##", out)
    out_incomplete = Symbol("##incomplete##", out)
    out_incompleteinds = Symbol("##incompleteinds##", out)
    push!(first_pass, :($out_incompleteinds = $out_incomplete.inds))
    push!(first_pass, :($out = $out_incomplete.data))
    seedout = Symbol("###seed###", out)
    seedout_missing = Symbol("###seed###", out_missing)
    tempstackptr = sptr isa Symbol ? Symbol("##temp##", sptr) : sptr
    isym = gensym(:i)
    if partial
        if isunbounded(B) # no transformation
            # we fully handle the partial here
            # therefore, we set partial = false
            # to prevent load_transformations
            # from doing so as well.
            partial = false 
            ptr_∂θ = gensym(:ptr_∂θ)
            ∂gather_quote = quote
                $ptr_∂θ = pointer($∂θ)
                @inbounds for $isym ∈ 1:$M
                    $m.VectorizationBase.store!($ptr_∂θ + ($isym - 1) * $(sizeof(T)), $seedout[$out_incompleteinds[$isym]])
                end
                $∂θ += $M
            end
        else
            if sptr isa Symbol
                # Do we need to increment the stack pointer?
                seedout_init_quote = quote
                    $tempstackptr = $sptr
                    $seedout_missing = $m.PaddedMatrices.PtrVector{$M,$T,$M,$M}(pointer($sptr, $T))
                    $sptr += $M
                end
            else
                seedout_init_quote = quote
                    $seedout_missing = $m.PaddedMatrices.MutableFixedSizePaddedVector{$M,$T,$M,$M}(undef)
                end
            end
            ∂gather_quote = quote
                @inbounds for $isym ∈ 1:$M
                    $seedout_missing[$ism] = $seedout[$out_incompleteinds[$isym]]
                end
            end
            push!(second_pass, seedout_init_quote)
        end
        push!(second_pass, ∂gather_quote)
    end
    load_transformations!(
        first_pass, second_pass, B, out_missing, Int[ M ],
        partial, logjac, sptr, m, θ, ∂θ, copyexport
    )
    scatter_quote = quote
        @inbounds for $isym ∈ 1:$M
            $out[$out_incompleteinds[$isym]] = $out_missing[$isym]
        end
    end
    push!(first_pass, scatter_quote)
    if partial && sptr isa Symbol
        push!(second_pass, :($sptr = $tempstackptr))
    end
#    push!(first_pass, :($out = $out_missing.data))
    nothing
end





