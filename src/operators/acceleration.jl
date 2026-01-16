mutable struct AccelerationTCICache
    tci::Union{Nothing,TCI.TensorCI2{ComplexF64}}
end
AccelerationTCICache() = AccelerationTCICache(nothing)

function acceleration_pivots(
    x_grid,
    kv_grid,
    Mv;
    x_lsb_first::Bool = false,
    kv_lsb_first::Bool = false,
    unfoldingscheme::Symbol = :interleaved,
)
    R = length(x_grid)

    xmin = quantics_to_origcoord(x_grid, fill(1, R))
    xmax = quantics_to_origcoord(x_grid, fill(2, R))
    xmid = 0.5 * (xmin + xmax)
    xs = [xmin, xmid, xmax]

    ks = [0, 2, Mv - 2, Mv - 1]

    pivots = Vector{Vector{Int}}()

    for x in xs
        q_x = origcoord_to_quantics(x_grid, x)
        q_x = maybe_reverse_bits(q_x, x_lsb_first)
        for k in ks
            q_kv = origcoord_to_quantics(kv_grid, k)
            q_kv = maybe_reverse_bits(q_kv, kv_lsb_first)
            push!(pivots, combine_bits(q_x, q_kv, unfoldingscheme))
        end
    end

    return pivots
end

function get_acceleration_mpo(
    dt::Real,
    Lv::Real,
    Mv::Int,
    x_grid,
    kv_grid,
    electric_field_tt::TCI.TensorTrain;
    tolerance::Real = 1e-12,
    k_cut::Real = 2^8,
    beta::Real = 2.0,
    x_lsb_first::Bool = false,
    kv_lsb_first::Bool = false,
    unfoldingscheme::Symbol = :interleaved,
    accel_cache::Union{Nothing,AccelerationTCICache}=nothing,
    reuse_strategy::Symbol=:resweep,
)
    R = length(x_grid)
    @assert length(kv_grid) == R
    localdims = fill(2, 2R)

    initial_pivots = acceleration_pivots(
        x_grid,
        kv_grid,
        Mv;
        x_lsb_first = x_lsb_first,
        kv_lsb_first = kv_lsb_first,
        unfoldingscheme = unfoldingscheme,
    )

    function kernel(q_bits::AbstractVector{Int})
        q_x, q_kv = split_bits(q_bits, R, unfoldingscheme)

        q_kv_aligned = maybe_reverse_bits(q_kv, kv_lsb_first)

        kv_orig = quantics_to_origcoord(kv_grid, q_kv_aligned)
        n_v = k_to_n(kv_orig, Mv)
        kv_phys = 2π * n_v / Lv

        E_val = electric_field_tt(q_x)

        phase = -E_val * kv_phys * dt
        return (1 + 1im * phase - 0.5 * phase^2) * Theta(n_v; beta = beta, k_cut = k_cut)
    end

    cached_kernel = TCI.CachedFunction{ComplexF64}(kernel, localdims)

    tci = build_acceleration_tci!(
        accel_cache,
        cached_kernel,
        localdims,
        initial_pivots;
        tolerance = tolerance,
        pivotsearch = :rook,
        reuse_strategy = reuse_strategy,
    )
    U_tt = TCI.TensorTrain(tci)

    return tt_to_mpo(U_tt)
end

function collect_final_pivots(tci::TCI.TensorCI2)
    n = length(tci)
    for b in 1:n-1
        if length(tci.Iset[b+1]) == length(tci.Jset[b])
            return [vcat(tci.Iset[b+1][k], tci.Jset[b][k]) for k in eachindex(tci.Jset[b])]
        end
    end
    error("Could not reconstruct pivot set from TCI.")
end

function resweep_acceleration_tci!(
    tci::TCI.TensorCI2{ComplexF64},
    kernel;
    tolerance::Real,
    pivotsearch::Symbol,
)
    pivots = collect_final_pivots(tci)
    if !isempty(pivots)
        maxsample = maximum(abs, (kernel(p) for p in pivots))
        tci.maxsamplevalue = max(maxsample, eps())
    else
        tci.maxsamplevalue = 1.0
    end
    TCI.optimize!(
        tci,
        kernel;
        tolerance = tolerance,
        maxbonddim = typemax(Int),
        maxiter = 50,
        pivotsearch = pivotsearch,
        sweepstrategy = :backandforth,
        maxnglobalpivot = 0,
        nsearchglobalpivot = 0,
        normalizeerror = false,
        strictlynested = false,
    )
    return tci
end

function build_acceleration_tci!(
    accel_cache::Union{Nothing,AccelerationTCICache},
    kernel,
    localdims,
    initial_pivots;
    tolerance::Real,
    pivotsearch::Symbol,
    reuse_strategy::Symbol,
)
    existing_tci = accel_cache === nothing ? nothing : accel_cache.tci

    tci = if reuse_strategy == :resweep && existing_tci !== nothing
        resweep_acceleration_tci!(
            existing_tci,
            kernel;
            tolerance = tolerance,
            pivotsearch = pivotsearch,
        )
    elseif reuse_strategy == :reuse_pivots && existing_tci !== nothing
        pivots = collect_final_pivots(existing_tci)
        TCI.crossinterpolate2(
            ComplexF64,
            kernel,
            localdims,
            pivots;
            tolerance = tolerance,
            pivotsearch = pivotsearch,
        )[1]
    else
        TCI.crossinterpolate2(
            ComplexF64,
            kernel,
            localdims,
            initial_pivots;
            tolerance = tolerance,
            pivotsearch = pivotsearch,
        )[1]
    end

    if accel_cache !== nothing
        accel_cache.tci = tci
    end

    return tci
end
