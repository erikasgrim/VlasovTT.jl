function get_poisson_mpo(
    Lx::Real,
    Mx::Int,
    kx_grid;
    tolerance::Real = 1e-12,
    eps0::Real = 1.0,
)
    poisson_kernel = kx -> begin
        nx = k_to_n(kx, Mx)
        if nx == 0
            return 0.0 + 0.0im
        else
            kx_phys = 2π * nx / Lx
            return (-1im / (eps0 * kx_phys))
        end
    end

    qtci, _, _ = quanticscrossinterpolate(
        ComplexF64,
        poisson_kernel,
        kx_grid;
        tolerance = tolerance,
    )
    tt = TCI.TensorTrain(qtci.tci)
    println("Poisson MPO ranks: ", TCI.rank(tt))
    return tt_to_mpo(tt)
end

function get_charge_density(psi::MPS; dv::Real=1.0)
    N = length(psi)
    N == 0 && return psi

    working = copy(ITensors.cpu(psi))
    for i in 2:2:N
        si = siteind(working, i)
        ones_vec = ITensor(si)
        for n in 1:dim(si)
            ones_vec[si => n] = 1.0
        end

        Ai = working[i] * ones_vec
        if i < N
            working[i + 1] = Ai * working[i + 1]
        else
            working[i - 1] = working[i - 1] * Ai
        end
        working[i] = ITensor(1.0)
    end

    odd_tensors = [working[i] for i in 1:N if isodd(i)]
    odd_tensors[1] *= dv
    return MPS(odd_tensors)
end

function ones_mps(sites)
    N = length(sites)
    N == 0 && return MPS()

    links = N > 1 ? [Index(1, "Link,l=$i") for i in 1:(N - 1)] : Index[]
    tensors = Vector{ITensor}(undef, N)

    for i in 1:N
        si = sites[i]

        inds_i = if N == 1
            (si,)
        elseif i == 1
            (links[i], si)
        elseif i == N
            (links[i - 1], si)
        else
            (links[i - 1], links[i], si)
        end

        Ti = ITensor(inds_i...)
        for n in 1:dim(si)
            if N == 1
                Ti[si => n] = 1.0
            elseif i == 1
                Ti[links[i] => 1, si => n] = 1.0
            elseif i == N
                Ti[links[i - 1] => 1, si => n] = 1.0
            else
                Ti[links[i - 1] => 1, links[i] => 1, si => n] = 1.0
            end
        end

        tensors[i] = Ti
    end

    return MPS(tensors)
end

function get_electric_field_mps(
    psi_mps::MPS,
    full_poisson_mpo::TCI.TensorTrain;
    dv::Real=1.0,
    tolerance=1e-8,
    maxrank=nothing,
    alg="naive",
)
    cd_mps = get_charge_density(psi_mps; dv)

    id_mps = ones_mps(siteinds(cd_mps)) # TODO: To not create a new MPS each time, pass in an identity MPS
    cd_mps = id_mps - cd_mps

    sites_cd_mps = siteinds(cd_mps)
    sites_poisson_mpo = [[prime(s,1), s] for s in sites_cd_mps]
    full_poisson_mpo_it = MPO(full_poisson_mpo; sites=sites_poisson_mpo)
    electric_field_mps = apply(full_poisson_mpo_it, cd_mps; alg=alg, truncate=true, maxdim=maxrank, cutoff=tolerance)

    return electric_field_mps
end
