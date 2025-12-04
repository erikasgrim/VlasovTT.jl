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