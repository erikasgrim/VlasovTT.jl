function acceleration_pivots(x_grid, kv_grid, Mv; lsb_first::Bool = false)
    R = length(x_grid)

    xmin = quantics_to_origcoord(x_grid, fill(1, R))
    xmax = quantics_to_origcoord(x_grid, fill(2, R))
    xmid = 0.5 * (xmin + xmax)
    xs = [xmin, xmid, xmax]

    ks = [0, 2, Mv - 2, Mv - 1]

    pivots = Vector{Vector{Int}}()

    for x in xs
        q_x = origcoord_to_quantics(x_grid, x)
        for k in ks
            q_kv = origcoord_to_quantics(kv_grid, k)
            q_kv = lsb_first ? reverse(q_kv) : q_kv
            push!(pivots, interleave_bits(q_x, q_kv))
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
    lsb_first::Bool = false,
)
    R = length(x_grid)
    @assert length(kv_grid) == R
    localdims = fill(2, 2R)

    initial_pivots = acceleration_pivots(x_grid, kv_grid, Mv; lsb_first = lsb_first)

    function kernel(q_bits::AbstractVector{Int})
        q_x = q_bits[1:2:2R]
        q_kv = q_bits[2:2:2R]

        q_kv_aligned = lsb_first ? reverse(q_kv) : q_kv

        kv_orig = quantics_to_origcoord(kv_grid, q_kv_aligned)
        n_v = k_to_n(kv_orig, Mv)
        kv_phys = 2π * n_v / Lv

        E_val = electric_field_tt(q_x)

        phase = -E_val * kv_phys * dt
        return (1 + 1im * phase - 0.5 * phase^2)
    end

    tci, _, _ = TCI.crossinterpolate2(
        ComplexF64,
        kernel,
        localdims,
        initial_pivots;
        tolerance = tolerance,
    )
    U_tt = TCI.TensorTrain(tci)

    return tt_to_mpo(U_tt)
end
