

# Base.CartesianIndices(A::MissingDataArray) = CartesianIndices(A.data)

function parameter_names(::Type{<:AbstractMissingDataArray{M}}, s::Symbol) where {M}
    ss = strip_hashtags(s) * "_missing_#"
    [ss * string(m) for m ∈ 1:M]
end
function parameter_names(A::AbstractMissingDataArray{M}, s::Symbol) where {M}
    out = Vector{String}(undef,M)
    # ci = CartesianIndices(A)
    inds = A.inds
    ss = strip_hashtags(s)*"["
    for m in 1:M
        cim = inds[m]
        ssm = ss * string(cim[1])
        for i in 2:length(cim)
            ssm *= "," * string(cim[i])
        end
        out[m] = ssm * "]"
    end
    out
end

PaddedMatrices.type_length(::AbstractMissingDataArray{M}) where {M} = M
PaddedMatrices.type_length(::Type{<:AbstractMissingDataArray{M}}) where {M} = M
PaddedMatrices.param_type_length(::AbstractMissingDataArray{M}) where {M} = M
PaddedMatrices.param_type_length(::Type{<:AbstractMissingDataArray{M}}) where {M} = M

"""
This function is not type stable.
"""
function maybe_missing(A::AA) where {T,N,AA <: AbstractArray{Union{Missing,T},N}}
    M = sum(ismissing, A)
    M == 0 && return convert(Array{T}, A)
    l, u = extrema(skipmissing(A))
    lb = l > 0 ? zero(T) : -typemax(T)
    ub = u < 0 ? zero(T) :  typemax(T)
    Threads.nthreads() > 1 ? convert(ThreadedMissingDataArray{M,Bounds(lb,ub)}, A) : convert(MissingDataArray{M,Bounds(lb,ub)}, A)
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
    MissingDataArray{M,B,T,N,typeof(data)}(
        data, findall(ismissing, A)
    )
end


function Base.convert(::Type{<:ThreadedMissingDataArray{M}}, A::AbstractArray{Union{Missing,T}}) where {M,T}
    convert(ThreadedMissingDataArray{M,Bounds(typemin(T),typemax(T))}, A)
end
function Base.convert(::Type{<:ThreadedMissingDataArray{M,B}}, A::AA) where {M,B,T,N,AA <: AbstractArray{Union{Missing,T},N}}
#    M = sum(ismissing, A)
    #    M == 0 &&
    data = similar(A, Float64)
    ptr_A = Base.unsafe_convert(Ptr{T}, pointer(A)) # is this necessary?
    LoopVectorization.@vvectorize for i ∈ eachindex(A)
        data[i] = ptr_A[i]
    end
    nthreads = Threads.nthreads()
    datav = Vector{tyepof(data)}(undef, nthreads)
    datav[1] = data
    Threads.@threads for n in 2:nthreads
        datav[n] = copy(data)
    end
    MissingDataArray{M,B,T,N,typeof(data)}(
        datav, findall(ismissing, A)
    )
end


function load_missing_as_vector!(
    first_pass, second_pass, out, ::Type{<:AbstractMissingDataArray{M,B,T}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, copyexport::Bool = false
) where {M,B,T}
    load_parameter!(
        first_pass, second_pass, out, RealVector{M,B,T}, partial, m, sptr, logjac, copyexport
    )
end


function load_parameter!(
    first_pass, second_pass, out, ::Type{MA}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, copyexport::Bool = false
) where {M,B,T,MA <: AbstractMissingDataArray{M,B,T}}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    out_missing = Symbol("##missing##", out)
    out_incomplete = Symbol("##incomplete##", out)
    out_incompleteinds = Symbol("##incompleteinds##", out)
    push!(first_pass, :($out_incompleteinds = $out_incomplete.inds))
    if MA <: ThreadedMissingDataArray
        push!(first_pass, :($out = $out_incomplete.data[Threads.threadid()]))
    else
        push!(first_pass, :($out = $out_incomplete.data))
    end
    seedout = Symbol("###seed###", out)
    seedoutextract = Symbol("###seed###extract", out)
    seedout_missing = Symbol("###seed###", out_missing)
    tempstackptr = sptr isa Symbol ? Symbol("##temp##", sptr) : sptr
    isym = gensym(:i)
    if partial
        push!(second_pass, :($seedoutextract = $m.DistributionParameters.extract($seedout)))
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
                    $m.VectorizationBase.store!($ptr_∂θ + ($isym - 1) * $(sizeof(T)), $seedoutextract[$out_incompleteinds[$isym]])
                end
                $∂θ += $M
            end
        else
            if sptr isa Symbol
                # Do we need to increment the stack pointer?
                seedout_init_quote = quote
                    $tempstackptr = $sptr
                    $seedout_missing = $m.PaddedMatrices.PtrVector{$M,$T,$M}(pointer($sptr, $T))
                    $sptr += $M
                end
            else
                seedout_init_quote = quote
                    $seedout_missing = $m.PaddedMatrices.FixedSizeVector{$M,$T,$M}(undef)
                end
            end
            ∂gather_quote = quote
                @inbounds for $isym ∈ 1:$M
                    $seedout_missing[$ism] = $seedoutextract[$out_incompleteinds[$isym]]
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
    nothing
end



