function interleave_bits(q_first::AbstractVector{Int}, q_second::AbstractVector{Int})
    R = length(q_first)
    @assert length(q_second) == R
    q = Vector{Int}(undef, 2R)
    for r in 1:R
        q[2r - 1] = q_first[r]
        q[2r] = q_second[r]
    end
    return q
end

function Theta(n; beta::Real = 2.0, k_cut::Real = 2^8)
    return 1 / (exp((abs(n) - k_cut) * beta) + 1)
end

function k_to_n(k, M)
    if k <= M ÷ 2
        return k
    else
        return k - M
    end
end

function n_to_k(n, M)
    if n >= 0
        return n
    else
        return n + M
    end
end

function build_kv_tt(Lv, M, kv_grid; tolerance::Real = 1e-12)

    function kv_kernel(kv)
        q_bits_normal = origcoord_to_quantics(kv_grid, kv)
        q_bits_reversed = reverse(q_bits_normal)
        kv_reversed = quantics_to_origcoord(kv_grid, q_bits_reversed)

        return 2pi * k_to_n(kv_reversed, M) / Lv #* Theta(k_to_n(kv_reversed, M); k_cut = 2^14, beta = 2.0)
    end

    qtci, rank, _ = quanticscrossinterpolate(
        Float64,
        kv_kernel,
        kv_grid;
        tolerance = tolerance,
    )
    return tt_to_mpo(TCI.TensorTrain(qtci.tci))
end


function tt_to_mpo(tt::TCI.TensorTrain)
    entry_type = eltype(tt.sitetensors[1])
    n_sites = length(tt)

    mpo_cores = Vector{Array{entry_type,4}}(undef, n_sites)
    for site in 1:n_sites
        mps_core = tt[site]
        s = size(mps_core)

        mpo_core = zeros(entry_type, s[1], s[2], s[2], s[3])
        for d in 1:s[2]
            mpo_core[:, d, d, :] = mps_core[:, d, :]
        end
        mpo_cores[site] = mpo_core
    end
    return TCI.TensorTrain(mpo_cores)
end
