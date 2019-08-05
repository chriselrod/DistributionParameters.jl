
struct RealFloat{B,T} <: Real end
struct RealArray{S,B,T,N,P,L} <: PaddedMatrices.AbstractMutableFixedSizePaddedArray{S,T,N,P,L} end
const RealVector{M,B,T,L} = RealArray{Tuple{M},B,T,1,L,L}
const RealMatrix{M,N,B,T,L} = RealArray{Tuple{M,N},B,T,2,M,L}


function load_parameter!(
    first_pass, second_pass, out, ::Type{RealFloat}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, copyexport::Bool = false
)
    T = Float64
    B = Bounds(typemin(T), typemax(T))
    
    load_parameter(
        first_pass, second_pass, out, RealFloat{B,T}, partial, m, sptr, logjac, copyexport
    )
end


function load_parameter!(
    first_pass, second_pass, out, ::Type{RealFloat{BT}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, copyexport::Bool = false
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

    load_parameter(
        first_pass, second_pass, out, RealFloat{B,T}, partial, m, sptr, logjac, copyexport
    )
end


function load_parameter!(
    first_pass, second_pass, out, ::Type{RealFloat{B,T}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, copyexport::Bool = false
) where {T, B}
    shape = Array{Int}(undef)
    shape[] = 1
    load_transformations!(
        first_pass, second_pass, B, out, shape,
        partial, logjac, sptr,
        m, Symbol("##θparameter##"), Symbol("##∂θparameter##"),
        copyexport
    )    
end

function load_parameter!(
    first_pass, second_pass, out, ::Type{RealArray{S}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, copyexport::Bool = false
) where {S}
    T = Float64
    B = Bounds(typemin(T), typemax(T))
    
    load_parameter(
        first_pass, second_pass, out, RealArray{S,B,T}, partial, m, sptr, logjac, copyexport
    )
end


function load_parameter!(
    first_pass, second_pass, out, ::Type{RealArray{S,BT}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, copyexport::Bool = false
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

    load_parameter(
        first_pass, second_pass, out, RealArray{S,B,T}, partial, m, sptr, logjac, copyexport
    )
end

function load_parameter!y(
    first_pass, second_pass, out, ::Type{RealArray{S,B,T}}, partial::Bool = false,
    m::Module = DistributionParameters, sptr::Union{Symbol,Nothing} = nothing, logjac::Bool = true, copyexport::Bool = false
) where {S,B,T}
    load_transformations!(
        first_pass, second_pass, B, out, Int[S.parameters...],
        partial, logjac, sptr,
        m, Symbol("##θparameter##"), Symbol("##∂θparameter##"),
        copyexport
    )    
end


#=

@inline unwrap(x) = x
@inline unwrap(d::LinearAlgebra.Diagonal) = d.diag

struct RealFloat{T} <: Real
    data::T
end
struct PositiveFloat{T} <: Real
    data::T
end
struct LowerBoundedFloat{LB,T} <: Real
    data::T
end
struct UpperBoundedFloat{UB,T} <: Real
    data::T
end
struct BoundedFloat{LB,UB,T} <: Real
    data::T
end
struct UnitFloat{T} <: Real
    data::T
end
const ScalarParameter{T} = Union{
    RealFloat{T},
    PositiveFloat{T},
    LowerBoundedFloat{T},
    UpperBoundedFloat{T},
    BoundedFloat{T},
    UnitFloat{T}
}

function load_parameter(
    first_pass, second_pass, out, ::Type{V}, partial::Bool = false,
    m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true
) where {M, V <: ScalarParameter}
    ## If element type of the vector is unspecified, we fill in Float64 here.
    load_parameter(first_pass, second_pass, out, V{Float64}, partial, m, sp, logjac)
end
function load_parameter(
    first_pass, second_pass, out, ::Type{RealFloat{T}}, partial::Bool = false,
    m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true
) where {T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    push!(first_pass, :($out = $m.VectorizationBase.load($θ); $θ += 1))
    if partial
        push!(second_pass, :($m.VectorizationBase.store!($∂θ, $(Symbol("###seed###", out)))))
        push!(second_pass, :($∂θ += 1))
    end
    nothing
end
function load_parameter(
    first_pass, second_pass, out, ::Type{PositiveFloat{T}}, partial::Bool = false,
    m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true
) where {T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    θᵢ = gensym(:θ)
    push!(first_pass, quote
        $θᵢ = $m.VectorizationBase.load($θ)
        $out = exp($θᵢ)
        $θ += 1
    end)
    logjac && push!(first_pass, :(target = vadd(target, $θᵢ)))
    if partial
        push!(second_pass, :($m.VectorizationBase.store!($∂θ, one($T) + $(Symbol("###seed###", out)) * $out)))
        push!(second_pass, :($∂θ += 1))
    end
    nothing
end
function load_parameter(
    first_pass, second_pass, out, ::Type{LowerBoundedFloat{LB,T}},
    partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true
) where {LB,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    θᵢ = gensym(:θ)
    expθᵢ = gensym(out)
    push!(first_pass, quote
        $θᵢ = $m.VectorizationBase.load($θ)
        $expθᵢ = exp($θᵢ)
        $out = $LB + $expθᵢ
        $θ += 1
    end)
    logjac && push!(first_pass, :(target = vadd(target, $θᵢ)))
    if partial
        push!(second_pass, :($m.VectorizationBase.store!($∂θ, one($T) + $(Symbol("###seed###", out)) * $expθᵢ)))
        push!(second_pass, :($∂θ += 1))
    end
    nothing
end
function load_parameter(
    first_pass, second_pass, out, ::Type{UpperBoundedFloat{UB,T}},
    partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true
) where {UB,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    θᵢ = gensym(:θ)
    expθᵢ = gensym(out)
    push!(first_pass, quote
        $θᵢ = $m.VectorizationBase.load($θ)
        $expθᵢ = exp($θᵢ)
        $out = $UB - $expθᵢ
        $θ += 1
    end)
    logjac && push!(first_pass, :(target = vadd(target, $θᵢ)))
    if partial
        push!(second_pass, :($m.VectorizationBase.store!($∂θ, one($T) - $(Symbol("###seed###", out)) * $expθᵢ)))
        push!(second_pass, :($∂θ += 1))
    end
    nothing
end
function load_parameter(
    first_pass, second_pass, out, ::Type{BoundedFloat{LB,UB,T}},
    partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true
) where {LB,UB,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    ninvlogitout = gensym(out)
    invlogitout = gensym(out)
    ∂invlogitout = gensym(out)
    push!(first_pass, quote
        $ninvlogitout = one($T) / (one($T) + exp($m.VectorizationBase.load($θ)))
        $invlogitout = one($T) - $ninvlogitout
        $∂invlogitout = $ninvlogitout * $invlogitout
        $out = $LB + $(UB-LB) * $invlogitout
        $θ += 1
    end)
    logjac && push!(first_pass.args, :(target = vadd(target, log($∂invlogitout))))
    if partial
        push!(second_pass, :($m.VectorizationBase.store!($∂θ, one($T) - 2*$invlogitout + $(Symbol("###seed###", out)) * $∂invlogitout * $(T(UB - LB)))))
        push!(second_pass, :($∂θ += 1))
    end
    nothing
end
function load_parameter(
    first_pass, second_pass, out, ::Type{UnitFloat{T}},
    partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true
) where {T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    ninvlogitout = gensym(out)
    ∂invlogitout = gensym(out)
    push!(first_pass, quote
        # $ninvlogitout = one($T) / (one($T) + exp($(T(0.5)) * $m.VectorizationBase.load($θ)))
        $ninvlogitout = one($T) / (one($T) + exp($m.VectorizationBase.load($θ)))
        $out = one($T) - $ninvlogitout
        $∂invlogitout = $ninvlogitout * $out
        $θ += 1
        # target += $(T(0.5)) * log($∂invlogitout) # + $(log(UB - LB)) # drop the constant term
        # target += log($∂invlogitout) # + $(log(UB - LB)) # drop the constant term
          end)
    logjac && push!(first_pass, :(target = vadd(target, log($∂invlogitout)) ))
    if partial
        # push!(second_pass, :($m.VectorizationBase.store!($∂θ, $(T(0.5)) - $out + $(T(0.5))*$(Symbol("###seed###", out)) * $∂invlogitout)))
        push!(second_pass, :($m.VectorizationBase.store!($∂θ, one($T) - 2*$out + $(Symbol("###seed###", out)) * $∂invlogitout)))
        push!(second_pass, :($∂θ += 1))
    end
    nothing
end


struct RealVector{M,T,P,L} <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{M,T,P,L}
    data::NTuple{L,T}
end
struct PositiveVector{M,T,P,L} <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{M,T,P,L}
    data::NTuple{L,T}
end
struct LowerBoundVector{M,LB,T,P,L} <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{M,T,P,L}
    data::NTuple{L,T}
end

Struct Upperboundvector{M,Ub,T,P,L} <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{M,T,P,L}
    data::NTuple{L,T}
end
struct BoundedVector{M,LB,UB,T,P,L} <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{M,T,P,L}
    data::NTuple{L,T}
end
struct UnitVector{M,T,P,L} <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{M,T,P,L}
    data::NTuple{L,T}
end
const VectorParameter{M,T,P,L} = Union{
    RealVector{M,T,P,L},
    PositiveVector{M,T,P,L},
    LowerBoundVector{M,LB,T,P,L} where {LB},
    UpperBoundVector{M,UB,T,P,L} where {UB},
    BoundedVector{M,LB,UB,T,P,L} where {LB, UB},
    UnitVector{M,T,P,L}
}

@generated function RealVector{M}(data::NTuple{L,T}) where {M,T,L}
    quote
        $(Expr(:meta,:inline))
        RealVector{$M,$T,$L,$L}(data)
    end
end
@generated function RealVector{M,T}(data::NTuple{L,T}) where {M,T,L}
    quote
        $(Expr(:meta,:inline))
        RealVector{$M,$T,$L,$L}(data)
    end
end
@inline RealVector(A::PaddedMatrices.AbstractFixedSizePaddedVector{M,T,P,L}) where {M,T,P,L} = RealVector{M,T,P,L}(A.data)
@generated function PositiveVector{M}(data::NTuple{L,T}) where {M,T,L}
    quote
        $(Expr(:meta,:inline))
        PositiveVector{$M,$T,$L,$L}(data)
    end
end
@generated function PositiveVector{M,T}(data::NTuple{L,T}) where {M,T,L}
    quote
        $(Expr(:meta,:inline))
        PositiveVector{$M,$T,$L,$L}(data)
    end
end
@inline PositiveVector(A::PaddedMatrices.AbstractFixedSizePaddedVector{M,T,P,L}) where {M,T,P,L} = PositiveVector{M,T,P,L}(A.data)
@generated function LowerBoundVector{M,LB}(data::NTuple{L,T}) where {M,T,L,LB}
    quote
        $(Expr(:meta,:inline))
        LowerBoundVector{$M,$LB,$T,$L,$L}(data)
    end
end
@generated function LowerBoundVector{M,LB,T}(data::NTuple{L,T}) where {M,T,L,LB}
    quote
        $(Expr(:meta,:inline))
        LowerBoundVector{$M,$LB,$T,$L,$L}(data)
    end
end
@inline LowerBoundVector{LB}(A::PaddedMatrices.AbstractFixedSizePaddedVector{M,T,P,L}) where {M,LB,T,P,L} = LowerBoundVector{M,LB,T,P,L}(A.data)
@generated function UpperBoundVector{M,UB}(data::NTuple{L,T}) where {M,T,L,UB}
    quote
        $(Expr(:meta,:inline))
        UpperBoundVector{$M,$UB,$T,$L,$L}(data)
    end
end
@generated function UpperBoundVector{M,UB,L}(data::NTuple{L,T}) where {M,T,L,UB}
    quote
        $(Expr(:meta,:inline))
        UpperBoundVector{$M,$UB,$T,$L,$L}(data)
    end
end
@inline UpperBoundVector{UB}(A::PaddedMatrices.AbstractFixedSizePaddedVector{M,T,P,L}) where {M,UB,T,P,L} = UpperBoundVector{M,UB,T,P,L}(A.data)
@generated function BoundedVector{M,LB,UB}(data::NTuple{L,T}) where {M,T,L,LB,UB}
    quote
        $(Expr(:meta,:inline))
        BoundedVector{$M,$LB,$UB,$T,$L,$L}(data)
    end
end
@generated function BoundedVector{M,LB,UB,T}(data::NTuple{L,T}) where {M,T,L,LB,UB}
    quote
        $(Expr(:meta,:inline))
        BoundedVector{$M,$LB,$UB,$T,$L,$L}(data)
    end
end
@inline BoundedVector{LB,UB}(A::PaddedMatrices.AbstractFixedSizePaddedVector{M,T,P,L}) where {M,LB,UB,T,P,L} = BoundedVector{M,LB,UB,T,P,L}(A.data)
@generated function UnitVector{M}(data::NTuple{L,T}) where {M,T,L}
    quote
        $(Expr(:meta,:inline))
        UnitVector{$M,$T,$L,$L}(data)
    end
end
@generated function UnitVector{M,T}(data::NTuple{L,T}) where {M,T,L}
    quote
        $(Expr(:meta,:inline))
        UnitVector{$M,$T,$L,$L}(data)
    end
end
@inline UnitVector(A::PaddedMatrices.AbstractFixedSizePaddedVector{M,T,P,L}) where {M,T,P,L} = UnitVector{M,T,P,L}(A.data)

function load_parameter(
    first_pass, second_pass, out, ::Type{V},
    partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true
) where {M, V <: VectorParameter{M}}
    ## If element type of the vector is unspecified, we fill in Float64 here.
    load_parameter(first_pass, second_pass, out, V{Float64}, partial, m, sp, logjac)
end
function load_parameter(
    first_pass, second_pass, out, ::Type{<: RealVector{M,T}},
    partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true#, constrain_only = true
) where {M,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
#    if sp == nothing
#        push!(first_pass, :($out = $m.SIMDPirates.vload(RealVector{$M,$T}, $θ); $θ += $M))
    #    else
    if logjac
        push!(first_pass, quote
              #             (sp, $out) = $m.PaddedVector{$M,$T}(
              $out = $m.PtrVector{$M,$T,$M,$M}(pointer($θ))
#              $out = PtrVector{$M,$T,$M,$M}(pointer($sp,$T))
#              $sp = $sp + $(M*sizeof(T))
              $θ += $M
              end)
    else#if sp isa Symbol
        isym = gensym(:i)
        push!(first_pass, quote
              #             (sp, $out) = $m.PaddedVector{$M,$T}(
#              $out = $m.PtrVector{$M,$T,$M,$M}(pointer($θ))
              $(sp isa Symbol ? :($out = $m.PtrVector{$M,$T,$M,$M}(pointer($sp,$T))) : :($out = $m.MutableFixedSizePaddedVector{$M,$T}(undef)))
              $(macroexpand(m, quote
                            LoopVectorization.@vvectorize $T for $isym ∈ 1:$M
                            $out[$isym] = $θ[$isym]
                            end
                            end))
              $(sp isa Symbol ? :($sp = $sp + $(M*sizeof(T))) : nothing)
              $θ += $M
              end)
    end
#    end
    if partial
        isym = gensym(:i)
        push!(second_pass, quote
              $(macroexpand(m, quote
                            LoopVectorization.@vvectorize $T for $isym ∈ 1:$M
                            $∂θ[$isym] = ($(Symbol("###seed###", out)))[$isym]
                            end
                            end))
            $∂θ += $M
        end)
    end
    nothing
end
function load_parameter(
    first_pass, second_pass, out, ::Type{<: RealVector{M}},
    partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true
) where {M}
    load_parameter(first_pass, second_pass, out, RealVector{M,Float64}, partial, m, sp, logjac)
end

function load_parameter(
    first_pass, second_pass, out, ::Type{<: PositiveVector{M,T}},
    partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true
) where {M,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    θᵢ = gensym(:θ)
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    V = Vec{W,T}
    sumθᵢ = gensym(:sumθᵢ)
    n_unmasked_loads = M >> Wshift
    Wm1 = W - 1
    rem = M & Wm1

    if sp isa Symbol
        #        push!(first_pass, :( ($sp,$out) = PtrVector{$M,$T}($sp)   ))
        isym = gensym(:i)
        temp = gensym()
        push!(first_pass, quote
              $out = PtrVector{$M,$T,$M,$M}(pointer($sp,$T))
              $sp = $sp + $(M*sizeof(T))
              $(macroexpand(m, quote
                            LoopVectorization.@vvectorize $T for $isym ∈ 1:$M
                            $temp = $θ[$isym]
                            $(logjac ? :(target = vadd(target, $temp)) : nothing)
                            $out[$isym] = exp($temp)
                            end
                            end))
              $θ += $M
              end)
    else
        outtup = Expr(:tuple,)
        if n_unmasked_loads > 0
            v = gensym(:v)
            push!(first_pass, :($v = $m.SIMDPirates.vload($V, $θ)))
            push!(first_pass, :($sumθᵢ = $v))
            push!(first_pass, :($v = $m.SLEEFPirates.exp($v)))
            for w ∈ 1:W
                push!(outtup.args, :($v[$w].value))
            end
            for n ∈ 1:n_unmasked_loads-1
                v = gensym(:v)
                push!(first_pass, :($v = $m.SIMDPirates.vload($V, $θ + $(n*W))))
                push!(first_pass, :($sumθᵢ = $m.SIMDPirates.vadd($sumθᵢ, $v)))
                push!(first_pass, :($v = $m.SLEEFPirates.exp($v)))
                for w ∈ 1:W
                    push!(outtup.args, :($v[$w].value))
                end
            end
        end
        L = (M + Wm1) & ~Wm1
        if rem != 0
            v = gensym(:v)
            push!(first_pass, :($v = $m.SIMDPirates.vload($V, $θ + $(n_unmasked_loads*W), $(unsafe_trunc(VectorizationBase.mask_type(W), 2^rem-1)))))
            if n_unmasked_loads > 0
                push!(first_pass, :($sumθᵢ = $m.SIMDPirates.vadd($sumθᵢ, $v)))
            else
                push!(first_pass, :($sumθᵢ = $v))
            end
            push!(first_pass, :($v = $m.SLEEFPirates.exp($v)))
            for w ∈ 1:W
                push!(outtup.args, :($v[$w].value))
            end
        end
        logjac && push!(first_pass, :(target = vadd(target, $sumθᵢ)))
        push!(first_pass, :($θ += $M))
        push!(first_pass, :($out = @inbounds PositiveVector{$M}($outtup)))
    end
    if partial
        i = gensym(:i)
        push!(second_pass, quote
              $(macroexpand(m, quote
                            LoopVectorization.@vvectorize $T for $i ∈ 1:$M
                            $∂θ[$i] = one($T) + ($(Symbol("###seed###", out)))[$i] * ($out)[$i]
                            end
                            end))
            $∂θ += $M
        end)
    end
    nothing
end
function load_parameter(
    first_pass, second_pass, out, ::Type{<: PositiveVector{M}},
    partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true
) where {M}
    load_parameter(first_pass, second_pass, out, PositiveVector{M,Float64}, partial, m, sp, logjac)
end
function load_parameter(
    first_pass, second_pass, out, ::Type{<: LowerBoundVector{M,LB,T}},
    partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true
) where {M,LB,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    θᵢ = gensym(:θ)
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    V = Vec{W,T}
    sumθᵢ = gensym(:sumθᵢ)
    n_unmasked_loads = M >> Wshift
    Wm1 = W - 1
    rem = M & Wm1

    vlb = gensym(:LB)
    if sp isa Symbol
        #        push!(first_pass, :( ($sp,$out) = PtrVector{$M,$T}($sp)   ))
        isym = gensym(:i)
        temp = gensym()
        push!(first_pass, quote
              $out = PtrVector{$M,$T,$M,$M}(pointer($sp,$T))
              $sp = $sp + $(M*sizeof(T))
              $(macroexpand(m, quote
                            LoopVectorization.@vvectorize $T for $isym ∈ 1:$M
                            $temp = $θ[$isym]
                            $(logjac ? :(target = vadd(target, $temp)) : nothing )
                            $out[$isym] = exp($temp) + $(T(LB))
                            end
                            end))
              $θ += $M
              end)
    else
        if n_unmasked_loads > 0
            v = gensym(:v)
            push!(first_pass, :($vlb = $m.SIMDPirates.vbroadcast($V, $(T(LB)))))
            push!(first_pass, :($v = $m.SIMDPirates.vload($V, $θ)))
            push!(first_pass, :(target = vadd(target, $v)))
            #        push!(first_pass, :($sumθᵢ = $v))
            push!(first_pass, :($v = $m.SIMDPirates.vadd($vlb, $m.SLEEFPirates.exp($v))))
            outtup = Expr(:tuple,)
            for w ∈ 1:W
                push!(outtup.args, :($v[$w].value))
            end
            for n ∈ 1:n_unmasked_loads-1
                v = gensym(:v)
                push!(first_pass, :($v = $m.SIMDPirates.vload($V, $θ + $(n*W))))
                push!(first_pass, :(target = vadd(target, $v)))
                #            push!(first_pass, :($sumθᵢ = $m.SIMDPirates.vadd($sumθᵢ, $v)))
                push!(first_pass, :($v = $m.SIMDPirates.vadd($vlb, $m.SLEEFPirates.exp($v))))
                for w ∈ 1:W
                    push!(outtup.args, :($v[$w].value))
                end
            end
        end
        L = (M + Wm1) & ~Wm1
        if rem != 0
            v = gensym(:v)
            push!(first_pass, :($v = $m.SIMDPirates.vload($V, $θ + $(n_unmasked_loads*W), $(unsafe_trunc(VectorizationBase.mask_type(W), 2^rem-1)))))
            push!(first_pass, :(target = vadd(target, $v)))
            #        if n_unmasked_loads > 0
            #            push!(first_pass, :($sumθᵢ = $m.SIMDPirates.vadd($sumθᵢ, $v)))
            #        else
            #            push!(first_pass, :($sumθᵢ = $v))
            #        end
            push!(first_pass, :($v = $m.SIMDPirates.vadd($vlb, SLEEFPirates.exp($v))))
            for w ∈ 1:W
                push!(outtup.args, :($v[$w].value))
            end
        end
        #    push!(first_pass, :(target += $m.SIMDPirates.vsum($sumθᵢ)))
        push!(first_pass, :($out = @inbounds LowerBoundVector{$M,$LB}($outtup)))
    end
    if partial
        i = gensym(:i)
        push!(second_pass, quote
              $(macroexpand(m, quote
                            LoopVectorization.@vvectorize $T for $i ∈ 1:$M
                            $∂θ[$i] = one($T) + ($(Symbol("###seed###", out)))[$i] * ($out)[$i]
                            end
                            end))
            $∂θ += $M
        end)
    end
    nothing
end
function load_parameter(first_pass, second_pass, out, ::Type{<: LowerBoundVector{M,LB}}, partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true) where {M,LB}
    load_parameter(first_pass, second_pass, out, LowerBoundVector{M,LB,Float64}, partial, m, sp, logjac)
end
function load_parameter(first_pass, second_pass, out, ::Type{<: UpperBoundVector{M,UB,T}}, partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true) where {M,UB,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    θᵢ = gensym(:θ)
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    V = Vec{W,T}
    sumθᵢ = gensym(:sumθᵢ)
    n_unmasked_loads = M >> Wshift
    Wm1 = W - 1
    rem = M & Wm1
    L = (M + Wm1) & ~Wm1

    if sp isa Symbol
        #        push!(first_pass, :( ($sp,$out) = PtrVector{$M,$T}($sp)   ))
        isym = gensym(:i)
        temp = gensym()
        push!(first_pass, quote
              $out = PtrVector{$M,$T,$M,$M}(pointer($sp,$T))
              $sp = $sp + $(M*sizeof(T))
              $(macroexpand(m, quote
                            LoopVectorization.@vvectorize $T for $isym ∈ 1:$M
                            $temp = $θ[$isym]
                            $(logjac ? :(target = vadd(target, $temp)) : nothing )
                            $out[$isym] = $(T(LB)) - exp($temp)
                            end
                            end))
              $θ += $M
              end)
    else
        vlb = gensym(:LB)
        if n_unmasked_loads > 0
            v = gensym(:v)
            push!(first_pass, :($vub = $m.SIMDPirates.vbroadcast($V, $(T(UB)))))
            push!(first_pass, :($v = $m.SIMDPirates.vload($V, $θ)))
            push!(first_pass, :(target = vadd(target, $v)))
            #       push!(first_pass, :($sumθᵢ = $v))
            push!(first_pass, :($v = $m.SIMDPirates.vsub($vub, $m.SLEEFPirates.exp($v))))
            outtup = Expr(:tuple,)
            for w ∈ 1:W
                push!(outtup.args, :($v[$w].value))
            end
            for n ∈ 1:n_unmasked_loads-1
                v = gensym(:v)
                push!(first_pass, :($v = $m.SIMDPirates.vload($V, $θ + $(n*W))))
                logjac && push!(first_pass, :(target = vadd(target, $v)))
                #            push!(first_pass, :($sumθᵢ = $m.SIMDPirates.vadd($sumθᵢ, $v)))
                push!(first_pass, :($v = $m.SIMDPirates.vsub($vub, $m.SLEEFPirates.exp($v))))
                for w ∈ 1:W
                    push!(outtup.args, :($v[$w].value))
                end
            end
        end
        if rem != 0
            v = gensym(:v)
            push!(first_pass, :($v = $m.SIMDPirates.vload($V, $θ + $(n_unmasked_loads*W), $(unsafe_trunc(VectorizationBase.mask_type(W), 2^rem-1)))))
            logjac && push!(first_pass, :(target = vadd(target, $v)))
            #        if n_unmasked_loads > 0
            #            push!(first_pass, :($sumθᵢ = $m.SIMDPirates.vadd($sumθᵢ, $v)))
            #        else
            #            push!(first_pass, :($sumθᵢ = $v))
            #        end
            push!(first_pass, :($v = $m.SIMDPirates.vsub($vub, $m.SLEEFPirates.exp($v))))
            for w ∈ 1:W
                push!(outtup.args, :($v[$w].value))
            end
        end
#    push!(first_pass, :(target += $m.SIMDPirates.vsum($sumθᵢ)))
        push!(first_pass, :($out = @inbounds UpperBoundVector{$M,$UB}($outtup)))
    end
    if partial
        i = gensym(:i)
        push!(second_pass, quote
              $(macroexpand(m, quote
                            LoopVectorization.@vvectorize $T for $i ∈ 1:$M
                            $∂θ[$i] = one($T) + ($(Symbol("###seed###", out)))[$i] * ($out)[$i]
                            end
                            end))
            $∂θ += $M
        end)
    end
    nothing
end
function load_parameter(first_pass, second_pass, out, ::Type{<: UpperBoundVector{M,UB}}, partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true) where {M,UB}
    load_parameter(first_pass, second_pass, out, UpperBoundVector{M,UB,Float64}, partial, m, sp, logjac)
end

function load_parameter(first_pass, second_pass, out, ::Type{<: BoundedVector{M,LB,UB,T}}, partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true) where {M,LB,UB,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    ninvlogitout = gensym(out)
    invlogitout = gensym(out)
    ∂invlogitout = gensym(out)
    mv = gensym(:mvector)

    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    sumθᵢ = gensym(:sumθᵢ)
    Wm1 = W - 1
    rem = M & Wm1
    L = (M + Wm1) & ~Wm1
    log_jac = gensym(:log_jac)
    i = gensym(:i)
    if sp isa Symbol
        q = quote
            $out = $m.PtrVector{$M,$T,$M,$M}(pointer($sp, $T))
            $sp = $sp + $(M*sizeof(T))
        end
    else
        q = quote $out = MutableFixedSizePaddedVector{$M,$T}(undef) end
    end
    push!(q.args, :($log_jac = $m.SIMDPirates.vbroadcast(Vec{$W,$T}, zero($T))))
    if partial
        if sp isa Symbol
            push!(q.args, :(($sp,$invlogitout) = $m.PtrVector{$M,$T}($sp)))
            push!(q.args, :(($sp,$∂invlogitout) = $m.PtrVector{$M,$T}($sp)))
        else
            push!(q.args, :($invlogitout = MutableFixedSizePaddedVector{$M,$T}(undef)))
            push!(q.args, :($∂invlogitout = MutableFixedSizePaddedVector{$M,$T}(undef)))
        end
    end
    loop_body = quote
        $ninvlogitout = one($T) / (one($T) + $m.SLEEFPirates.exp($θ[$i]))
    end
    if partial
        push!(loop_body.args, :($invlogitout[$i] = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout[$i] = $ninvlogitout * $invlogitout[$i]))
        push!(loop_body.args, :($out[$i] = $LB + $(UB-LB) * $invlogitout[$i]))
        logjac && push!(loop_body.args, :(target = vadd(target, $m.SLEEFPirates.log($∂invlogitout[$i]))))
    else
        push!(loop_body.args, :($invlogitout = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout = $ninvlogitout * $invlogitout))
        push!(loop_body.args, :($out[$i] = $LB + $(UB-LB) * $invlogitout))
        logjac && push!(loop_body.args, :(target = vadd(target, $m.SLEEFPirates.log($∂invlogitout))))
    end

    push!(q.args, quote
          $(macroexpand(m, quote
                        LoopVectorization.@vvectorize $T for i ∈ 1:$M
                        $loop_body
                        end
                        end))
#        $out = BoundedVector{$M,$LB,$UB}($mv)
        $θ += $M
#        target += $m.SIMDPirates.vsum($log_jac)
    end)
    push!(first_pass, q)
    if partial
        push!(second_pass, quote
              $(macroexpand(m, quote
                            LoopVectorization.@vvectorize $T for $i ∈ 1:$M
                            $∂θ[$i] = one($T) - 2($invlogitout)[$i] + ($(Symbol("###seed###", out)))[$i] * ($∂invlogitout)[$i] * $(T(UB - LB))
                            end
                            end))
            $∂θ += $M
        end)
    end
    nothing
end
function load_parameter(first_pass, second_pass, out, ::Type{<: BoundedVector{M,LB,UB}}, partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true) where {M,LB,UB}
    load_parameter(first_pass, second_pass, out, BoundedVector{M,LB,UB,Float64}, partial, m, sp, logjac)
end

function load_parameter(first_pass, second_pass, out, ::Type{<: UnitVector{M,T}}, partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true) where {M,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    ninvlogitout = gensym(out)
    invlogitout = gensym(out)
    ∂invlogitout = gensym(out)
  #  mv = gensym(:mvector)
#    mv = out
    i = gensym(:i)
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    sumθᵢ = gensym(:sumθᵢ)
    Wm1 = W - 1
    rem = M & Wm1
    L = (M + Wm1) & ~Wm1
#    log_jac = gensym(:log_jac)
    if sp isa Symbol
        q = quote
            $out = $m.PtrVector{$M,$T,$M,$M}(pointer($sp, $T))
            $sp = $sp + $(M*sizeof(T))
        end
    else
        q = quote $out = MutableFixedSizePaddedVector{$M,$T}(undef) end
    end
#    push!(q.args, :($log_jac = $m.SIMDPirates.vbroadcast(SIMDPirates.Vec{$W,$T}, zero($T)) ))
    if partial
        if sp isa Symbol
            push!(q.args, :(($sp,$invlogitout) = $m.PtrVector{$M,$T}($sp)))
            push!(q.args, :(($sp,$∂invlogitout) = $m.PtrVector{$M,$T}($sp)))
        else
            push!(q.args, :($invlogitout = MutableFixedSizePaddedVector{$M,$T}(undef)))
            push!(q.args, :($∂invlogitout = MutableFixedSizePaddedVector{$M,$T}(undef)))
        end
    end
    loop_body = quote
        $ninvlogitout = one($T) / (one($T) + $m.SLEEFPirates.exp($θ[$i]))
    end
    if partial
        push!(loop_body.args, :($invlogitout[$i] = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout[$i] = $ninvlogitout * $invlogitout[$i]))
        push!(loop_body.args, :($out[$i] = $invlogitout[$i]))
        logjac && push!(loop_body.args, :(target = vadd(target, $m.SLEEFPirates.log($∂invlogitout[$i]))))
    else
        push!(loop_body.args, :($invlogitout = one($T) - $ninvlogitout))
        push!(loop_body.args, :($∂invlogitout = $ninvlogitout * $invlogitout))
        push!(loop_body.args, :($out[$i] = $invlogitout))
        logjac && push!(loop_body.args, :(target = vadd(target, $m.SLEEFPirates.log($∂invlogitout))))
    end

    push!(q.args, quote
          $(macroexpand(m, quote
                        LoopVectorization.@vvectorize $T for $i ∈ 1:$M
                        $loop_body
                        end
                        end))
          #        $out = UnitVector{$M}($mv)
  #        $out = $mv
        $θ += $M
#        target += $m.SIMDPirates.vsum($log_jac)
    end)
    push!(first_pass, q)
    if partial
        push!(second_pass, quote
              $(macroexpand(m, quote
                            LoopVectorization.@vvectorize $T for $i ∈ 1:$M
                            $∂θ[$i] = one($T) - 2($invlogitout)[$i] + ($(Symbol("###seed###", out)))[$i] * ($∂invlogitout)[$i]
                            end
                            end))
            $∂θ += $M
        end)
    end
    nothing
end
function load_parameter(first_pass, second_pass, out, ::Type{<: UnitVector{M}}, partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true) where {M}
    load_parameter(first_pass, second_pass, out, UnitVector{M,Float64}, partial, m, sp, logjac)
end


struct RealMatrix{M,N,T,P,L} <: PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L}
    data::NTuple{L,T}
end
struct PositiveMatrix{M,N,T,P,L} <: PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L}
    data::NTuple{L,T}
end
struct LowerBoundMatrix{M,N,LB,T,P,L} <: PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L}
    data::NTuple{L,T}
end
struct UpperBoundMatrix{M,N,UB,T,P,L} <: PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L}
    data::NTuple{L,T}
end
struct BoundedMatrix{M,N,LB,UB,T,P,L} <: PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L}
    data::NTuple{L,T}
end
struct UnitMatrix{M,N,T,P,L} <: PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L}
    data::NTuple{L,T}
end
const MatrixParameter{M,N,T,P,L} = Union{
    RealMatrix{M,N,T,P,L},
    PositiveMatrix{M,N,T,P,L},
    LowerBoundMatrix{M,N,LB,T,P,L} where {LB},
    UpperBoundMatrix{M,N,UB,T,P,L} where {UB},
    BoundedMatrix{M,N,LB,UB,T,P,L} where {LB, UB},
    UnitMatrix{M,N,T,P,L}
}

@generated function RealMatrix{M,N}(data::NTuple{L,T}) where {M,N,T,L}
    P = L ÷ N
    quote
        $(Expr(:meta,:inline))
        RealMatrix{$M,$N,$T,$P,$L}(data)
    end
end
@generated function RealMatrix{M,N,T}(data::NTuple{L,T}) where {M,N,T,L}
    P = L ÷ N
    quote
        $(Expr(:meta,:inline))
        RealMatrix{$M,$N,$T,$P,$L}(data)
    end
end
@generated function PositiveMatrix{M,N}(data::NTuple{L,T}) where {M,N,T,L}
    P = L ÷ N
    quote
        $(Expr(:meta,:inline))
        RealMatrix{$M,$N,$T,$P,$L}(data)
    end
end
@generated function PositiveMatrix{M,N,T}(data::NTuple{L,T}) where {M,N,T,L}
    P = L ÷ N
    quote
        $(Expr(:meta,:inline))
        RealMatrix{$M,$N,$T,$P,$L}(data)
    end
end
@generated function LowerBoundMatrix{M,N,LB}(data::NTuple{L,T}) where {M,N,T,L,LB}
    P = L ÷ N
    quote
        $(Expr(:meta,:inline))
        RealMatrix{$M,$N,$LB,$T,$P,$L}(data)
    end
end
@generated function LowerBoundMatrix{M,N,LB,T}(data::NTuple{L,T}) where {M,N,T,L,LB}
    P = L ÷ N
    quote
        $(Expr(:meta,:inline))
        RealMatrix{$M,$N,$LB,$T,$P,$L}(data)
    end
end
@generated function UpperBoundMatrix{M,N,UB}(data::NTuple{L,T}) where {M,N,T,L,UB}
    P = L ÷ N
    quote
        $(Expr(:meta,:inline))
        RealMatrix{$M,$N,$UB,$T,$P,$L}(data)
    end
end
@generated function UpperBoundMatrix{M,N,UB,T}(data::NTuple{L,T}) where {M,N,T,L,UB}
    P = L ÷ N
    quote
        $(Expr(:meta,:inline))
        RealMatrix{$M,$N,$UB,$T,$P,$L}(data)
    end
end
@generated function BoundedMatrix{M,N,LB,UB}(data::NTuple{L,T}) where {M,N,T,L,LB,UB}
    P = L ÷ N
    quote
        $(Expr(:meta,:inline))
        RealMatrix{$M,$N,$LB,$UB,$T,$P,$L}(data)
    end
end
@generated function BoundedMatrix{M,N,LB,UB,T}(data::NTuple{L,T}) where {M,N,T,L,LB,UB}
    P = L ÷ N
    quote
        $(Expr(:meta,:inline))
        RealMatrix{$M,$N,$LB,$UB,$T,$P,$L}(data)
    end
end
@generated function UnitMatrix{M,N}(data::NTuple{L,T}) where {M,N,T,L}
    P = L ÷ N
    quote
        $(Expr(:meta,:inline))
        RealMatrix{$M,$N,$T,$P,$L}(data)
    end
end
@generated function UnitMatrix{M,N,T}(data::NTuple{L,T}) where {M,N,T,L}
    P = L ÷ N
    quote
        $(Expr(:meta,:inline))
        RealMatrix{$M,$N,$T,$P,$L}(data)
    end
end


function load_parameter(first_pass, second_pass, out, ::Type{<: RealMatrix{M,N,T}}, partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true) where {M,N,T}
    θ = Symbol("##θparameter##")
    ∂θ = Symbol("##∂θparameter##")
    push!(first_pass, :($out = $m.PaddedMatrices.PtrMatrix{$M,$N,$T,$M}(pointer($θ)) ))# $m.SIMDPirates.vload(RealMatrix{M,N,T}, $θ); $θ += $(M*N)))
    push!(first_pass, :($θ += $(M*N)))
    if partial
        i = gensym(:i)
        push!(second_pass, quote
            for n ∈ 0:$(N-1)
              $m.LoopVectorization.@vvectorize $T for $i ∈ 1:$M
                    $∂θ[$i+$M*n] = ($(Symbol("###seed###", out)))[$i+$M*n]
                end
            end
            $∂θ += $(M*N)
        end)
    end
    nothing
end
function load_parameter(first_pass, second_pass, out, ::Type{A}, partial::Bool = false, m::Module = DistributionParameters, sp::Union{Symbol,Nothing} = nothing, logjac::Bool = true) where {M, N, A <: MatrixParameter{M,N}}
    ## If element type of the vector is unspecified, we fill in Float64 here.
    load_parameter(first_pass, second_pass, out, A{Float64}, partial, m, sp, logjac)
end


const Parameters{T} = Union{
    ScalarParameter{T},
    VectorParameter{L,T} where {L},
    MatrixParameter{M,N,T} where {M,N}
}
const PositiveParameter{T} = Union{
    PositiveFloat{T},
    PositiveVector{L,T} where {L},
    PositiveMatrix{M,N,T} where {M,N}
}
const LowerBoundedParameter{T,LB} = Union{
    LowerBoundedFloat{LB,T},
    LowerBoundVector{L,LB,T} where {L},
    LowerBoundMatrix{M,N,LB,T} where {M,N}
}
const UpperBoundedParameter{T,UB} = Union{
    UpperBoundedFloat{UB,T},
    UpperBoundVector{L,UB,T} where {L},
    UpperBoundMatrix{M,N,UB,T} where {M,N}
}
const BoundedParameter{T,LB,UB} = Union{
    BoundedFloat{LB,UB,T},
    BoundedVector{L,LB,UB,T} where {L},
    BoundedMatrix{M,N,LB,UB,T} where {M,N}
}
const UnitParameter{T} = Union{
    UnitFloat{T},
    UnitVector{L,T} where {L},
    UnitMatrix{M,N,T} where {M,N}
}


isparameter(::T) where {T <: Parameters} = true
isparameter(::Type{T}) where {T <: Parameters} = true
isparameter(::Any) = false

ispositive(::T) where {T <: PositiveParameter} = true
ispositive(::Type{T}) where {T <: PositiveParameter} = true
ispositive(::Any) = false

islowerbounded(::T) where {T <: LowerBoundedParameter} = true
islowerbounded(::Type{T}) where {T <: LowerBoundedParameter} = true
islowerbounded(::Any) = false

isupperbounded(::T) where {T <: UpperBoundedParameter} = true
isupperbounded(::Type{T}) where {T <: UpperBoundedParameter} = true
isupperbounded(::Any) = false

isbounded(::T) where {T <: BoundedParameter} = true
isbounded(::Type{T}) where {T <: BoundedParameter} = true
isbounded(::Any) = false

isunit(::T) where {T <: UnitParameter} = true
isunit(::Type{T}) where {T <: UnitParameter} = false
isunit(::Any) = false

bounds(::LowerBoundedParameter{T,LB}) where {T,LB} = (T(LB),T(Inf))
bounds(::UpperBoundedParameter{T,UB}) where {T,UB} = (T(-Inf),T(UB))
bounds(::BoundedParameter{T,LB,UB}) where {T,LB,UB} = (T(LB), T(UB))
bounds(::Any) = (-Inf,Inf)

lower_bound(::Type{LowerBoundedParameter{T,LB}}) where {T,LB} = LB
upper_bound(::Type{UpperBoundedParameter{T,UB}}) where {T,UB} = UB
bounds(::Type{BoundedParameter{T,LB,UB}}) where {T,LB,UB} = (LB, UB)
=#

PaddedMatrices.type_length(::Type{<:ScalarParameter}) = 1


strip_hashtags(s::String) = s[1+last(findlast("#",s)):end]
strip_hashtags(s::Symbol) = strip_hashtags(string(s))
parameter_names(::Type{<:ScalarParameter}, s::Symbol) = [strip_hashtags(s)]
function parameter_names(::Type{<: VectorParameter{M}}, s::Symbol) where {M}
    ss = strip_hashtags(s)
    ["$ss[$m]" for m ∈ 1:M]::Vector{String}
end
function parameter_names(::Type{<: MatrixParameter{M,N}}, s::Symbol) where {M,N}
    ss = strip_hashtags(s)
    vec(["$ss[$m,$n]" for m ∈ 1:M, n ∈ 1:N])
end


# type_length(::Type{<:PaddedMatrices.AbstractFixedSizePaddedVector{N}}) where {N} = N
# type_length(::Type{<:PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{M,N}}) where {M,N} = M*N

# complete_type(v) = v
# has_length(::Type{<:VectorParameter{M}}) where {M} = true
# has_length(::Type{<:VectorParameter}) = false
# has_dims(::Type{<:MatrixParameter{M,N}}) where {M,N} = true
# has_dims(::Type{<:MatrixParameter}) = false
# function complete_type(::Type{V}) where {M,T,V<:VectorParameter{M,T}}
#     N,P,L = PaddedMatrices.calc_NPL(Tuple{M}, T)
#     V2 = V{P}
#     isa(V2, UnionAll) || return Val{V2}()
#     V3 = V2{L}
#     isa(V3, UnionAll) || return Val{V3}()
#     throw("Don't know how to complete type $V.")
# end
# function complete_type(::Type{A}) where {M,N,T,A<:MatrixParameter{M,N,T}}
#     D,P,L = PaddedMatrices.calc_NPL(Tuple{M,N}, T)
#     A2 = A{P,L}
#     isa(A2, UnionAll) || return Val{A2}()
#     throw("Don't know how to complete type $A.")
# end
#
#
# @generated function complete_type(::Type{T}) where {T}
#     isa(T, UnionAll) || return Val{T}()
#     if T <: ScalarParameter
#         if T <: ScalarParameter{<:Union{Float64,Float32}}
#             throw("Don't know how to complete parameter $T.")
#         else
#             T2 = T{Float64}
#             if isa(T2, UnionAll)
#                 throw("Don't know how to complete parameter $T2.")
#             else
#                 return Val{T2}()
#             end
#         end
#     elseif T <: VectorParameter
#         if !has_length(T)
#             throw("Don't know how to infer length for parameter $T.")
#         end
#         if T <: VectorParameter{M,<:Union{Float32,Float64}} where {M}
#             T2 = T
#         else
#             T2 = T{Float64}
#             # isa(T2, UnionAll) || return T2
#         end
#         return complete_type(T2)
#     elseif T <: MatrixParameter
#         if !has_dims(T)
#             throw("Don't know how to infer length for parameter $T.")
#         end
#         if T <: MatrixParameter{M,N,<:Union{Float32,Float64}} where {M,N}
#             T2 = T
#         else
#             T2 = T{Float64}
#             # isa(T2, UnionAll) || return T2
#         end
#         return complete_type(T2)
#     end
# end




