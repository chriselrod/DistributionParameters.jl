


#=
abstract type Constraint end
struct LowerBound{LB} <: Constraint end
struct UpperBound{UB} <: Constraint end
struct Positive <: Constraint end
struct Unit <: Constraint end
struct Bounded{LB,UB} <: Constraint end
=#



"""
Using bounds as a parameteric type will force generated functions to recompile.
However, the expression manipulation functions at least will not have to via
making all Bounds a single type, and using if/else checks to determine bound type.
"""
struct Bounds{T <: Real}
    lb::T
    ub::T
end

Bounds() = Bounds(-Inf,Inf)

ismin(x::T) = ((x == typemin(T)) | (x == -floatmin(T)))
ismax(x::T) = ((x == typemax(T)) | (x ==  floatmin(T)))
function isunbounded(b::Bounds{T}) where {T}
    ismin(b.lb) & ismax(b.ub)
end
function islowerbound(b::Bounds{T}) where {T}
    (!ismin(b.lb)) & ismax(b.ub)
end
function isupperbound(b::Bounds{T}) where {T}
    ismin(b.lb) & (!ismax(b.ub))
end
function isbounded(b::Bounds{T}) where {T}
    (!ismin(b.lb)) & (!ismax(b.ub))
end
#=
function ispositive(b::Bounds{T}) where {T}
    (b.lb == zero(T)) & (b.ub == typemax(T))
end
function isnegative(b::Bounds{T}) where {T}
    (b.lb == typemin(T)) & (b.ub == zero(T))
end
function isunit(b::Bounds{T}) where {T}
    (b.lb == zero(T)) & (b.ub == one(T))
end
=#
#drop_jacobian(b::Bounds) = isbounded(b)


"""
Function applies the transformations for loading a parameter to
fp = (quote end).args
sp = (quote end).args
M == param_length == 0 => scalar
"""
function load_transformations!(
    fp, sp, b::Bounds{T}, out, shape::Vector{Int},
    partial::Bool = true, logjac::Bool = true, sptr,
    m::Module = DistributionParameters,
    θ = Symbol("##θparameter##"), ∂θ = Symbol("##∂θparameter##"),
    copyexport::Bool = false
) where {T}
    N = length(shape)
    scalar = iszero(N)
    M = prod(shape)

#    plincr = scalar ? 1 : M
    outinit = if sptr isa Symbol
        quote
            $out = $m.PtrArray{$(Tuple{shape...}),$T,$N,$(first(shape)),$M,true}(pointer($sptr, $T))
            $sptr += $(sizeof(T)*M)
        end
    else
        quote # Do we want to pad these?
            $out = $m.MutableFixedSizePaddedArray{$(Tuple{shape...}),$T,$N,$(first(shape)),$M}(undef)
        end
    end
    seedout = Symbol("###seed###", out)
    loopmacro = Symbol("@vvectorize")
    if isunbounded(b)
        if scalar
            push!(fp, :($out = $m.VectorizationBase.load($θ)))
        elseif copyexport
            isym = gensym(:i)
            loop_quote = quote
                LoopVectorization.@vvectorize $T for $isym ∈ 1:$M
                    $out[$isym] = $θ[$isym]
                end
            end
            copy_q = quote
                $outinit
                $(macroexpand(m, loop_quote))
            end
            push!(fp, copy_q)
        else
            push!(fp, :($out = $m.PtrArray{$(Tuple{shape...}),$T,$N,$(first(shape)),$M,true}(pointer($θ))))
        end
        if partial
            if scalar
                push!(sp, :($m.VectorizationBase.store!($∂θ, $seedout)))
            else
                isym = gensym(:i)
                storeloop = Expr(:macrocall, loopmacro, LineNumberNode(@__LINE__, @__FILE__),
                                 Expr(:for, Expr(:(=), isym, pstorerange),
                                      :($∂θ[$isym] = $seedout[$isym])
                                      )
                                 )
#                storeloop = quote
#                   LoopVectorization.@vvectorize $T for $isym ∈ 1:$M
#                        $∂θ[$isym] = $seedout[$isym]
#                    end
#                end
                push!(sp, macroexpand(m, storeloop))
            end
        end
#    elseif ispositive(b)
    elseif islowerbounded(b)
        logout = gensym(Symbol(:log_, out))
        outdef = (b.lb == zero(T)) ? :(exp($logout)) : :(exp($logout) + $(T(b.lb)))
        if scalar
            load_expr = quote
                $logout = $m.VectorizationBase.load($θ)
                $out = $outdef
            end
            logjac && push!(load_expr, :(target = $m.SIMDPirates.vadd(target, $logout)))
        else
            isym = gensym(:i)
            loopbody = quote
                $logout = $θ[$isym]
                $out[$isym] = $outdef
            end
            logjac && push!(loopbody, :(target = SIMDPirates.vadd(target, $logout)))
            loop_quote = quote
                LoopVectorization.@vvectorize $T for $isym ∈ 1:$M
                    $loopbody
                end
            end
            load_expr = quote
                $outinit
                $(macroexpand(m, loop_quote))
            end
        end
        push!(fp, loop_expr)
        if partial
            if scalar
                push!(sp, :($m.VectorizationBase.store!($∂θ, muladd($seedout, $out, one($T)))))
            else
                isym = gensym(:i)
                storeloop = Expr(:macrocall, loopmacro, LineNumberNode(@__LINE__, @__FILE__),
                                 Expr(:for, Expr(:(=), isym, pstorerange),
                                      :($∂θ[$isym] = SIMDPirates.vmuladd($seedout[$isym], $out[$isym], one($T)))
                                      )
                                 )
#                storeloop = quote
#                    LoopVectorization.@vvectorize $T for $isym ∈ 1:$M
#                        $∂θ[$isym] = SIMDPirates.vmuladd($seedout[$isym], $out[$isym], one($T))
#                    end
#                end
                push!(sp, macroexpand(m, storeloop))
            end
        end
    elseif isupperbounded(b)
        logout = gensym(Symbol(:log_, out))
        outdef = (b.ub == zero(T)) ? :(- exp($logout)) : :($(T(b.ub)) - exp($logout))
        if scalar
           n load_expr = quote
                $logout = $m.VectorizationBase.load($θ)
                $out = $outdef
            end
            logjac && push!(load_expr, :(target = $m.SIMDPirates.vadd(target, $logout)))
        else
            isym = gensym(:i)
            loopbody = quote
                $logout = $θ[$isym]
                $out[$isym] = $outdef
            end
            logjac && push!(loopbody, :(target = SIMDPirates.vadd(target, $logout)))
            loop_quote = quote
                LoopVectorization.@vvectorize $T for $isym ∈ 1:$M
                    $loopbody
                end
            end
            load_expr = quote
                $outinit
                $(macroexpand(m, loop_quote))
            end
        end
        push!(fp, loop_expr)
        if partial
            if scalar
                push!(sp, :($m.VectorizationBase.store!($∂θ, SIMDPirates.vfnmadd($seedout, $out, one($T)))))
            else
                isym = gensym(:i)
                storeloop = Expr(:macrocall, loopmacro, LineNumberNode(@__LINE__, @__FILE__),
                                 Expr(:for, Expr(:(=), isym, pstorerange),
                                      :($∂θ[$isym] = SIMDPirates.vfnmadd($seedout[$isym], $out[$isym], one($T)))
                                      )
                                 )
                push!(sp, macroexpand(m, storeloop))
            end
        end
    elseif isbounded(b)
        scale = b.ub - b.lb
        invlogit = gensym(:invlogit)
        ninvlogit = gensym(:ninvlogit)
        ∂invlogit = gensym(:∂invlogit)
#        outdef = ispositive(b) ? :(- exp($logout)) : :($(T(b.ub)) - exp($logout))
        if scalar
            q = quote
                $ninvlogit = one($T) / (one($T) + exp($m.VectorizationBase.load($θ)))
                $invlogit = one($T) - $ninvlogit
                $out = $(b.lb == zero(T) ?
                         (scale == one(T) ? invlogit : :($scale * $invlogit)) :
                         (scale == one(T) ? :($(b.lb) + $invlogit) : :(muladd($scale, $invlogit, $(b.lb)))))
                $∂invlogit = $invlogit * $ninvlogit
            end
            logjac && push!(q.args, :( target = SIMDPirates.vadd(target, $∂invlogit)))
            push!(fp, q)
            if partial
                ∂q = quote
                    VectorizationBase.store!($∂θ, @fastmath one($T) - $(T(2)) * $invlogit +
                                             $seedout * $(scale == one(T) ? ∂invlogit : :($scale * $∂invlogit)))
                end
                push!(sp, macroexpand(m, ∂q)) #Expand the fastmath? Why not Base.FastMath.add_fast / mul_fast directly?
            end
        else
            isym = gensym(:i)
            if partial
                invlogitinits = if sp isa Symbol
                    quote
                        $invlogit = $m.PtrVector{$M,$T}(pointer($sptr,$T))
                        $sptr += $(M*sizeof(T))
                        $∂invlogit = $m.PtrVector{$M,$T}(pointer($sptr,$T))
                        $sptr += $(M*sizeof(T))
                    end
                else
                    quote
                        $invlogit = MutableFixedSizePaddedVector{$M,$T}(undef)
                        $∂invlogit = MutableFixedSizePaddedVector{$M,$T}(undef)))
                    end
                end
                push!(fp, invlogitinits)
                ilt = gensym(:ilt)
                ∂ilt = gensym(:∂ilt)
                loop_body = quote
                    $ninvlogit = one($T) / (one($T) + $m.SLEEFPirates.exp($θ[$isym]))
                    $ilt = one($T) - $ninvlogit
                    $invlogit[$isym] = $ilt
                    $∂ilt = $ilt * $ninvlogit
                    $∂invlogit[$isym] = $∂ilt
                    $out[$isym] = $(b.lb == zero(T) ?
                                    (scale == one(T) ? ilt : :($scale * $ilt)) :
                                    (scale == one(T) ? :($(b.lb) + $ilt) : :(SIMDPirates.vmuladd($scale, $ilt, $(b.lb)))))
                end
                logjac && push!(loop_body.args, :(target = vadd(target, $m.SLLEFPirates.log($∂ilt))))
                loop_quote = quote
                    LoopVectorization.@vvectorize $T for $isym ∈ 1:$M
                        $loop_body
                    end
                end
                push!(fp, macroexpand(m, loop_quote))
                storeloop = Expr(:macrocall, loopmacro, LineNumberNode(@__LINE__, @__FILE__),
                                 Expr(:for, Expr(:(=), isym, pstorerange),
                                      :($∂θ[$isym] = one($T) - $(T(2)) * $invlogit[$isym] +
                                        ($seedout)[$isym] *
                                        $(scale == one(T) ? :($∂invlogit[$isym]) : :($∂invlogit[$isym] * $scale))
                                        )
                                      )
                                 )
                push!(sp, macroexpand(m, storeloop))
            else
                loop_body = quote
                    $ninvlogit = one($T) / (one($T) + $m.SLEEFPirates.exp($θ[$isym]))
                    $invlogit = one($T) - $ninvlogit
                    $out[$isym] = $(b.lb == zero(T) ?
                                    (scale == one(T) ? invlogit : :($scale * $invlogit)) :
                                    (scale == one(T) ? :($(b.lb) + $invlogit) : :(SIMDPirates.vmuladd($scale, $invlogit, $(b.lb)))))
                end
                logjac && push!(loop_body.args, :(target = vadd(target, $m.SLLEFPirates.log($invlogit * $ninvlogit))))
                loop_quote = quote
                    LoopVectorization.@vvectorize $T for $isym ∈ 1:$M
                        $loop_body
                    end
                end
                push!(fp, macroexpand(m, loop_quote))
            end
        end
    end
    push!(fp, :($θ += $M))
    partial && push!(sp, :($∂θ += $M))
end

#=
function jacobian_expr(b::Bounds{T}) where {T}
end
=#


