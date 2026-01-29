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

function interleave_bits(q_first::AbstractVector{<:AbstractVector{Int}}, q_second::AbstractVector{<:AbstractVector{Int}})
    @assert length(q_first) == length(q_second)
    return [interleave_bits(q_first[i], q_second[i]) for i in eachindex(q_first)]
end

function combine_bits(
    q_first::AbstractVector{Int},
    q_second::AbstractVector{Int},
    unfoldingscheme::Symbol,
)
    if unfoldingscheme == :interleaved
        return interleave_bits(q_first, q_second)
    elseif unfoldingscheme == :separate
        return vcat(q_first, q_second)
    end
    error("Unsupported unfoldingscheme: $(unfoldingscheme)")
end

function combine_bits(
    q_first::AbstractVector{<:AbstractVector{Int}},
    q_second::AbstractVector{<:AbstractVector{Int}},
    unfoldingscheme::Symbol,
)
    @assert length(q_first) == length(q_second)
    return [combine_bits(q_first[i], q_second[i], unfoldingscheme) for i in eachindex(q_first)]
end

function maybe_reverse_bits(bits::AbstractVector{Int}, lsb_first::Bool)
    return lsb_first ? reverse(bits) : bits
end

function split_interleaved_bits(q_bits::AbstractVector{Int}, R::Int)
    return q_bits[1:2:2R], q_bits[2:2:2R]
end

function split_bits(q_bits::AbstractVector{Int}, R::Int, unfoldingscheme::Symbol)
    if unfoldingscheme == :interleaved
        return split_interleaved_bits(q_bits, R)
    elseif unfoldingscheme == :separate
        return q_bits[1:R], q_bits[(R + 1):(2R)]
    end
    error("Unsupported unfoldingscheme: $(unfoldingscheme)")
end

function frequency_filter(n; beta::Real = 2.0, k_cut::Real = 2^8)
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

function abs2_mps(psi_mps::MPS; alg::String="naive", maxdim=nothing, cutoff=1e-8)
    psi = ITensors.cpu(psi_mps)
    N = length(psi)
    N == 0 && return psi

    sites = siteinds(psi)
    orthogonalize!(psi, 1)

    link_dims = N > 1 ? [dim(linkind(psi, i)) for i in 1:(N - 1)] : Int[]
    new_links = N > 1 ? [Index(link_dims[i]^2, "Link,l=$(i)") for i in 1:(N - 1)] : Index[]

    tensors = Vector{ITensor}(undef, N)
    for i in 1:N
        s = sites[i]

        if i == 1
            r = linkind(psi, i)
            A_arr = Array(psi[i], s, r)
            C_arr = Array{eltype(A_arr)}(undef, dim(s), dim(r)^2)
            for sv in 1:dim(s)
                slice = view(A_arr, sv, :)
                C_arr[sv, :] = vec(LinearAlgebra.kron(slice, conj(slice)))
            end
            tensors[i] = ITensor(C_arr, s, new_links[i])
        elseif i == N
            l = linkind(psi, i - 1)
            A_arr = Array(psi[i], l, s)
            C_arr = Array{eltype(A_arr)}(undef, dim(l)^2, dim(s))
            for sv in 1:dim(s)
                slice = view(A_arr, :, sv)
                C_arr[:, sv] = vec(LinearAlgebra.kron(slice, conj(slice)))
            end
            tensors[i] = ITensor(C_arr, new_links[i - 1], s)
        else
            l = linkind(psi, i - 1)
            r = linkind(psi, i)
            A_arr = Array(psi[i], l, s, r)
            C_arr = Array{eltype(A_arr)}(undef, dim(l)^2, dim(s), dim(r)^2)
            for sv in 1:dim(s)
                slice = view(A_arr, :, sv, :)
                C_arr[:, sv, :] = LinearAlgebra.kron(slice, conj(slice))
            end
            tensors[i] = ITensor(C_arr, new_links[i - 1], s, new_links[i])
        end
    end

    if maxdim !== nothing && N > 1
        for i in 2:N
            left_inds = uniqueinds(tensors[i - 1], tensors[i])
            phi = tensors[i - 1] * tensors[i]
            U, S, V = svd(phi, left_inds; maxdim = maxdim, cutoff = cutoff)
            tensors[i - 1] = U
            tensors[i] = S * V
        end
    end

    f_mps = MPS(tensors)
    if maxdim !== nothing
        truncate!(f_mps; maxdim = maxdim, cutoff = cutoff)
    end
    return f_mps
end
