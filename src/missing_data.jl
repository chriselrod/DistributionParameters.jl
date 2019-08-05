

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

"""
This function is not type stable.
"""
function maybe_missing(A::AA) where {T,N,AA <: AbstractArray{Union{Missing,T},N}}
    M = sum(ismissing, A)
    M == 0 && return A
    l, u = extrema(skipmissing(AA))
    lb = l < 0 ? -typemax(T) : zero(T)
    ub = u > 0 ? typemax(T) : zero(T)
    convert(MissingDataArray{M,Bounds(lb,ub)}, A)
end

function Base.convert(::Type{<:MissingDataArray{M}}, A::AbstractArray{T}) where {M,T}
    convert(MissingDataArray{M,Bounds(typemin(T),typemax(T))}, A)
end
function Base.convert(::Type{<:MissingDataArray{M,B}}, A::AA) where {T,B,N,AA <: AbstractArray{Union{Missing,T},N}}
#    M = sum(ismissing, A)
    #    M == 0 &&
    data = similar(A)
    ptr_A = Base.unsafe_convert(Ptr{T}, pointer(A)) # is this necessary?
    LoopVectorization.@vvectorize for i ∈ eachindex(A)
        data[i] = ptr_A[i]
    end
    inds = findall(ismissing, A)
    MissingDataArray{M,B,T,N,AA}(
        data, inds
    )
end


function load_parameter!(
    first_pass, second_pass, out, ::Type{MissingDataArray{M,B,T}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, copyexport::Bool = false
) where {M,B,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    out_missing = Symbol("##missing##", out)
    out_incomplete = Symbol("##incomplete##", out)
    out_incompleteinds = Symbol("##incompleteinds##", out)
    push!(first_pass.args, :($out_incompleteinds = $out_incomplete.inds))
    push!(first_pass.args, :($out = $out_incomplete.data))
    seedout = Symbol("###seed###", out)
    seedout_missing = Symbol("###seed###", out_missing)
    tempstackptr = sptr isa Symbol ? Symbol("##temp##", sptr) : sptr
    if partial
        isym = gensym(:i)
        if isunbounded(B) # no transformation
            # we fully handle the partial here
            # therefore, we set partial = false
            # to prevent load_transformations
            # from doing so as well.
            partial = false 
            ∂gather_quote = quote
                @inbounds for $isym ∈ 1:$M
                    $∂θ[$isym] = $seedout[$out_incompleteinds[$isym]]
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
    complete_out_quote = quote
        @inbounds for $isym ∈ 1:$M
            $out[$out_incompleteinds[$isym]] = $out_missing[$isym]
        end
    end
    if partial && sptr isa Symbol
        push!(second_pass, :($sptr = $tempstackptr))
    end
#    push!(first_pass, :($out = $out_missing.data))
    nothing
end




