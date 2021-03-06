



# """
# Function applies the transformations for loading a parameter to
# fp = (quote end).args
# sp = (quote end).args
# M == param_length == 0 => scalar
# """
function load_transformations!(
    fp, sp, b::Bounds{T}, out, shape::Vector{Int},
    partial::Bool, logjac::Bool, sptr::Union{Symbol,Nothing},
    m::Module = DistributionParameters,
    θ = Symbol("##θparameter##"), ∂θ = Symbol("##∂θparameter##"),
    exportparam::Bool = false
) where {T}
    maybe_align = x -> exportparam ? x : VectorizationBase.align(x)
    N = length(shape)
    scalar = iszero(N)
    M = prod(shape)
    if !scalar
        X = similar(shape); X[1] = 1
        for n in 2:N
            X[n] = X[n-1] * shape[n-1]
        end
    end
    use_sptr = sptr isa Symbol
    outinit = if scalar
        quote end
    elseif use_sptr
        if N == 1
            quote
                $out = $m.PtrVector{$(first(shape)),$T}(pointer($sptr, $T))
                $sptr += $(maybe_align(sizeof(T)*M))
            end
        else
            quote
                $out = $m.PtrArray{$(Tuple{shape...}),$T,$N,$(Tuple{X...}),$M,true}(pointer($sptr, $T))
                $sptr += $(maybe_align(sizeof(T)*M))
            end
        end
    else
        quote # Do we want to pad these?
            $out = $m.FixedSizeArray{$(Tuple{shape...}),$T,$N,$(Tuple{X...}),$M}(undef)
        end
    end
    outlifestart = if !scalar && !exportparam && use_sptr
        push!(outinit.args, :($m.lifetime_start!($out)))
        true
    else
        false
    end
    adjout = ReverseDiffExpressionsBase.adj(out)
    if isunbounded(b)
        if scalar
            push!(fp, :($out = $m.VectorizationBase.load($θ)))
        elseif exportparam
            isym = gensym(:i)
            loop_quote = quote
                LoopVectorization.@vvectorize_unsafe $T $m for $isym ∈ 1:$M
                    $out[$isym] = $θ[$isym]
                end
            end
            copy_q = quote
                $outinit
                $(macroexpand(m, loop_quote))
            end
            push!(fp, copy_q)
        else
            push!(fp, :($out = $m.PtrArray{$(Tuple{shape...}),$T,$N,$(Tuple{X...}),$M,true}(pointer($θ))))
        end
    elseif islowerbounded(b)
        logout = gensym(Symbol(:log_, out))
        if scalar
            outdef = ((b.lb == zero(T)) && !exportparam ) ? :(RealFloat{$(Bounds(zero(T),typemax(T))),$T,$T}(Base.exp($logout),$logout)) : :(exp($logout) + $(T(b.lb)))
            load_expr = quote
                $logout = $m.VectorizationBase.load($θ)
                $out = $outdef
            end
            logjac && push!(load_expr.args, :(target = $m.vadd(target, $logout)))
        else
            outdef = (b.lb == zero(T)) ? :(exp($logout)) : :(exp($logout) + $(T(b.lb)))
            isym = gensym(:i)
            loopbody = quote
                $logout = $θ[$isym]
                $out[$isym] = $outdef
            end
            logjac && push!(loopbody.args, :(target = $m.vadd(target, $logout)))
            loop_quote = quote
                LoopVectorization.@vvectorize_unsafe $T $m for $isym ∈ 1:$M
                    $loopbody
                end
            end
            if b.lb == zero(T) && !exportparam && sptr !== nothing
                outinit = quote
                    $out = $m.RealArray{$(Tuple{shape...}),$(Bounds(zero(T),typemax(T))),$T,$N,$(Tuple{X...}),$M,Ptr{$T}}(pointer($sptr, $T),pointer($θ))
                    $sptr += $(maybe_align(sizeof(T)*M))
                end
            end
            load_expr = quote
                $outinit
                $(macroexpand(m, loop_quote))
            end
        end
        push!(fp, load_expr)
        # (exportparam || scalar) || push!(fp, load_expr)
        if partial
            if scalar
                # push!(sp, :($m.VectorizationBase.store!($∂θ, muladd($adjout, $out, one($T)))))
                push!(sp, :($m.VectorizationBase.store!($adjout, muladd($m.VectorizationBase.load($adjout), $out, one($T)))))
            else
                isym = gensym(:i)
                storeloop = quote
                    LoopVectorization.@vvectorize_unsafe $T $m for $isym ∈ 1:$M
                        $adjout[$isym] = $m.SIMDPirates.vmuladd($adjout[$isym], $out[$isym], one($T))
                    end
                end
                push!(sp, macroexpand(m, storeloop))
            end
        end
        outlifestart && push!(sp, :($m.lifetime_end!($out)))
    elseif isupperbounded(b)
        logout = gensym(Symbol(:log_, out))
        outdef = ((b.ub == zero(T)) && !exportparam) ? :(- exp($logout)) : :($(T(b.ub)) - exp($logout))
        if scalar
            load_expr = quote
                $logout = $m.VectorizationBase.load($θ)
                $out = $outdef
            end
            logjac && push!(load_expr.args, :(target = $m.vadd(target, $logout)))
        else
            isym = gensym(:i)
            loopbody = quote
                $logout = $θ[$isym]
                $out[$isym] = $outdef
            end
            logjac && push!(loopbody.args, :(target = $m.vadd(target, $logout)))
            loop_quote = quote
                LoopVectorization.@vvectorize_unsafe $T $m for $isym ∈ 1:$M
                    $loopbody
                end
            end
            load_expr = quote
                $outinit
                $(macroexpand(m, loop_quote))
            end
        end
        push!(fp, load_expr)
        if partial
            if scalar
                push!(sp, :($m.VectorizationBase.store!($adjout, $m.SIMDPirates.vfnmadd($m.VectorizationBase.load($adjout), $out, one($T)))))
            else
                isym = gensym(:i)
                storeloop = quote
                    LoopVectorization.@vvectorize_unsafe $T $m for $isym ∈ 1:$M
                        $adjout[$isym] = $m.SIMDPirates.vfnmadd($adjout[$isym], $out[$isym], one($T))
                    end
                end
                push!(sp, macroexpand(m, storeloop))
            end
        end
        outlifestart && push!(sp, :($m.lifetime_end!($out)))
    elseif isbounded(b)
        scale = b.ub - b.lb
        invlogit = gensym(:invlogit)
        ninvlogit = gensym(:ninvlogit)
        ∂invlogit = gensym(:∂invlogit)
        if scalar
            if b.lb == 0 && b.ub == 1
                unconstrained = gensym(:unconstrained)
                q = quote
                    $unconstrained = $m.VectorizationBase.load($θ)
                    $ninvlogit = one($T) / (one($T) + exp($unconstrained))
                    $invlogit = one($T) - $ninvlogit
                    $out = RealFloat{$(Bounds(zero(T),one(T))),$T,$T}($invlogit, $unconstrained)
                    $∂invlogit = $invlogit * $ninvlogit
                end
            else
                q = quote
                    $ninvlogit = one($T) / (one($T) + exp($m.VectorizationBase.load($θ)))
                    $invlogit = one($T) - $ninvlogit
                    $out = $(b.lb == zero(T) ?
                             (scale == one(T) ? invlogit : :($scale * $invlogit)) :
                             (scale == one(T) ? :($(b.lb) + $invlogit) : :(muladd($scale, $invlogit, $(b.lb)))))
                    $∂invlogit = $invlogit * $ninvlogit
                end
            end
            logjac && push!(q.args, :( target = $m.vadd(target, $∂invlogit)))
            push!(fp, q)
            if partial
                ∂q = quote
                    VectorizationBase.store!(
                        $adjout, @fastmath one($T) - $(T(2)) * $invlogit +
                        $m.VectorizationBase.load($adjout) * $(scale == one(T) ? ∂invlogit : :($scale * $∂invlogit))
                    )
                end
                push!(sp, macroexpand(m, ∂q)) #Expand the fastmath? Why not Base.FastMath.add_fast / mul_fast directly?
            end
        else
            isym = gensym(:i)
            if b.lb == 0 && b.ub == 1
                outinit = quote
                    $out = $m.RealArray{$(Tuple{shape...}),$(Bounds(zero(T),one(T))),$T,$N,$(Tuple{X...}),$M,Ptr{$T}}(pointer($sptr, $T), pointer($θ))
                    $sptr += $(maybe_align(sizeof(T)*M))
                end
                outlifestart && push!(outinit.args, :($m.lifetime_start!($out)))
            end
            if partial
                invlogitinits = if use_sptr
                    quote
                        $outinit
                        $invlogit = $m.PtrVector{$M,$T}(pointer($sptr,$T))
                        $sptr += $(maybe_align(M*sizeof(T)))
                        $∂invlogit = $m.PtrVector{$M,$T}(pointer($sptr,$T))
                        $sptr += $(maybe_align(M*sizeof(T)))
                    end
                else
                    quote
                        $outinit
                        $invlogit = FixedSizeVector{$M,$T}(undef)
                        $∂invlogit = FixedSizeVector{$M,$T}(undef)
                    end
                end
                if outlifestart
                    push!(invlogitinits.args, :($m.lifetime_start!($invlogit)))
                    push!(invlogitinits.args, :($m.lifetime_start!($∂invlogit)))
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
                                    (scale == one(T) ? :($(b.lb) + $ilt) : :($m.SIMDPirates.vmuladd($scale, $ilt, $(b.lb)))))
                end
                logjac && push!(loop_body.args, :(target = $m.vadd(target, $m.SLEEFPirates.log($∂ilt))))
                loop_quote = quote
                    LoopVectorization.@vvectorize_unsafe $T $((m)) for $isym ∈ 1:$M
                        $loop_body
                    end
                end
                push!(fp, macroexpand(m, loop_quote))
                storeloop = quote
                    LoopVectorization.@vvectorize_unsafe $T $((m)) for $isym ∈ 1:$M
                        $adjout[$isym] = one($T) - $(T(2)) * $invlogit[$isym] +
                            ($adjout)[$isym] * $(scale == one(T) ? :($∂invlogit[$isym]) : :($∂invlogit[$isym] * $scale))
                    end
                end
                push!(sp, macroexpand(m, storeloop))
                if outlifestart
                    push!(sp, :($m.lifetime_end!($invlogit)))
                    push!(sp, :($m.lifetime_end!($∂invlogit)))
                end
            else
                loop_body = quote
                    $ninvlogit = one($T) / (one($T) + $m.SLEEFPirates.exp($θ[$isym]))
                    $invlogit = one($T) - $ninvlogit
                    $out[$isym] = $(b.lb == zero(T) ?
                                    (scale == one(T) ? invlogit : :($scale * $invlogit)) :
                                    (scale == one(T) ? :($(b.lb) + $invlogit) : :($m.SIMDPirates.vmuladd($scale, $invlogit, $(b.lb)))))
                end
                logjac && push!(loop_body.args, :(target = $m.vadd(target, $m.SLEEFPirates.log($invlogit * $ninvlogit))))
                loop_quote = quote
                    $outinit
                    LoopVectorization.@vvectorize_unsafe $T $((m)) for $isym ∈ 1:$M
                        $loop_body
                    end
                end
                push!(fp, macroexpand(m, loop_quote))
            end
            outlifestart && push!(sp, :($m.lifetime_end!($out)))
        end
    end
    if partial
        if scalar
            push!(fp, :($adjout = $∂θ))
        else
            push!(fp, :($adjout = $m.PtrArray{$(Tuple{shape...}),$T,$N,$(Tuple{X...}),$M,true}(pointer($∂θ))))
        end
    end
    if exportparam && scalar
        push!(fp, :($m.VectorizationBase.store!(pointer($sptr, $T), convert($T, $out)); $sptr += $(sizeof(T))))
    end
    push!(fp, :($θ += $M))
    partial && push!(fp, :($∂θ += $M))
    nothing
end



