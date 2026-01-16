function free_streaming_pivots(
    kx_grid,
    v_grid,
    Mx;
    kx_lsb_first::Bool = false,
    v_lsb_first::Bool = false,
    unfoldingscheme::Symbol = :interleaved,
)
    R = length(kx_grid)

    ks = [0, 2, Mx - 2, Mx - 1]

    vmin = quantics_to_origcoord(v_grid, fill(1, R))
    vmax = quantics_to_origcoord(v_grid, fill(2, R))
    vmid = 0.5 * (vmin + vmax)
    vs = [vmin, vmid, vmax]

    pivots = Vector{Vector{Int}}()

    for k in ks
        q_kx = origcoord_to_quantics(kx_grid, k)
        q_kx = maybe_reverse_bits(q_kx, kx_lsb_first)
        for v in vs
            q_v = origcoord_to_quantics(v_grid, v)
            q_v = maybe_reverse_bits(q_v, v_lsb_first)
            push!(pivots, combine_bits(q_kx, q_v, unfoldingscheme))
        end
    end

    return pivots
end

function get_free_streaming_mpo(
    dt::Real,
    Lx::Real,
    Mx::Int,
    kx_grid,
    v_grid;
    tolerance::Real = 1e-12,
    k_cut::Real = 2^8,
    beta::Real = 2.0,
    kx_lsb_first::Bool = false,
    v_lsb_first::Bool = false,
    unfoldingscheme::Symbol = :interleaved,
)
    R = length(kx_grid)
    localdims = fill(2, 2R)

    initial_pivots = free_streaming_pivots(
        kx_grid,
        v_grid,
        Mx;
        kx_lsb_first = kx_lsb_first,
        v_lsb_first = v_lsb_first,
        unfoldingscheme = unfoldingscheme,
    )

    function kernel(q_bits::AbstractVector{Int})
        q_kx, q_v = split_bits(q_bits, R, unfoldingscheme)
        q_kx_aligned = maybe_reverse_bits(q_kx, kx_lsb_first)
        q_v_aligned = maybe_reverse_bits(q_v, v_lsb_first)

        kx_orig = quantics_to_origcoord(kx_grid, q_kx_aligned)
        v_orig = quantics_to_origcoord(v_grid, q_v_aligned)

        n_x = k_to_n(kx_orig, Mx)
        kx_phys = 2π * n_x / Lx
        
        return exp(-1im * kx_phys * v_orig * dt) * frequency_filter(n_x; beta = beta, k_cut = k_cut)
    end

    tci, _, _ = TCI.crossinterpolate2(
        ComplexF64,
        kernel,
        localdims,
        initial_pivots;
        tolerance = tolerance,
    )
    println("Free streaming MPO ranks: ", TCI.rank(tci))
    U_tt = TCI.TensorTrain(tci)

    return tt_to_mpo(U_tt)
end
