Base.@kwdef struct SimulationParams
    dt::Float64
    tolerance::Float64 = 1e-8
    cutoff::Float64 = 1e-8
    maxrank::Union{Int,Nothing} = nothing
    maxrank_ef::Union{Int,Nothing} = nothing
    k_cut::Real = 2^6
    beta::Real = 2.0
    v0::Real = 0.0
    use_gpu::Bool = false
    alg::String = "naive"
    l1_normalize::Bool = true
end

function strang_step_filtered_TCI!(
    psi_mps::MPS,
    phase::PhaseSpaceGrids,
    full_poisson_mpo::TCI.TensorTrain,
    free_stream_mpo_it::MPO,
    x_fourier_mpo_it::MPO,
    x_inv_fourier_mpo_it::MPO,
    v_fourier_mpo_it::MPO,
    v_inv_fourier_mpo_it::MPO,
    half_filtering_mpo_it::MPO,
    mps_sites,
    previous_tci;
    params::SimulationParams,
)

    # Get electric field MPO
    _, electric_field_mps = get_electric_field_mps_kv(
        psi_mps,
        full_poisson_mpo;
        dv = sqrt(phase.M) * phase.dv,
        cutoff = params.cutoff,
        maxrank_ef = params.maxrank_ef,
        alg = params.alg,
    )

    electric_field_tt = TCI.TensorTrain(ITensors.cpu(electric_field_mps))
    psi_tt = TCI.TensorTrain(ITensors.cpu(psi_mps))

    R = phase.R
    localdims = fill(2, 2 * phase.R)

    function kernel(q_bits::AbstractVector{Int})
        q_x, q_kv = split_interleaved_bits(q_bits, R)
        q_kv_aligned = maybe_reverse_bits(q_kv, phase.kv_lsb_first)

        kv_orig = quantics_to_origcoord(phase.kv_grid, q_kv_aligned)
        n_v = k_to_n(kv_orig, phase.M)
        kv_phys = 2π * n_v / phase.Lv

        E_val = electric_field_tt(q_x)

        phase_angle = E_val * kv_phys * params.dt
        return exp(im * phase_angle) * psi_tt(q_bits) * frequency_filter(kv_phys; beta=params.beta, k_cut=params.k_cut)
    end
    kernel = TCI.ThreadedBatchEvaluator{ComplexF64}(kernel, localdims)
    kernel = TCI.CachedFunction{ComplexF64}(kernel, localdims)

    # if previous_tci === nothing
    #     pivots = acceleration_pivots(phase.x_grid, phase.kv_grid, phase.M; lsb_first=true)
    # else
    #     pivots = collect_final_pivots(previous_tci)
    #     println(pivots)
    # end

    # psi_tt = TCI.crossinterpolate2(
    #     ComplexF64,
    #     kernel,
    #     fill(2, 2*phase.R),
    #     pivots;
    #     tolerance = params.tolerance,
    #     maxbonddim = params.maxrank,
    #     verbosity = 1
    # )[1]

    if previous_tci === nothing
        pivots = acceleration_pivots(
            phase.x_grid,
            phase.kv_grid,
            phase.M;
            x_lsb_first = phase.x_lsb_first,
            kv_lsb_first = phase.kv_lsb_first,
        )

        psi_tt = TCI.crossinterpolate2(
            ComplexF64,
            kernel,
            localdims,
            pivots;
            tolerance = params.tolerance,
            maxbonddim = params.maxrank,
        )[1]
    else
        TCI.optimize!(
            previous_tci,
            kernel;
            tolerance = params.tolerance,
            maxbonddim = params.maxrank,
            verbosity = 0,
        )
        psi_tt = previous_tci
    end

    psi_mps = MPS(psi_tt; sites = mps_sites)
    if params.use_gpu
        psi_mps = cu(psi_mps)
    end

    # Fourier transform in x
    psi_mps = apply(x_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Filtering 
    psi_mps = apply(half_filtering_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Inverse Fourier transform in v
    psi_mps = apply(v_inv_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Free streaming step
    psi_mps = apply(free_stream_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Fourier transform in v
    psi_mps = apply(v_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Filtering
    psi_mps = apply(half_filtering_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Inverse Fourier transform in x
    psi_mps = apply(x_inv_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    return psi_mps, electric_field_mps, psi_tt
end

function strang_step_unfiltered_TCI!(
    psi_mps::MPS,
    phase::PhaseSpaceGrids,
    full_poisson_mpo::TCI.TensorTrain,
    free_stream_mpo_it::MPO,
    x_fourier_mpo_it::MPO,
    x_inv_fourier_mpo_it::MPO,
    v_fourier_mpo_it::MPO,
    v_inv_fourier_mpo_it::MPO,
    x_inv_v_fourier_mpo_it::MPO,
    v_inv_x_fourier_mpo_it::MPO,
    mps_sites,
    previous_tci;
    params::SimulationParams,
    tci_profile::Bool = true,
    tci_profile_label::String = "",
)   
    steptime_start = time()

    # Get electric field MPO
    _, electric_field_mps = get_electric_field_mps_kv(
        psi_mps,
        full_poisson_mpo;
        dv = sqrt(phase.M) * phase.dv,
        cutoff = params.cutoff,
        maxrank_ef = params.maxrank_ef,
        alg = params.alg,
    )

    electric_field_tt = TCI.TensorTrain(ITensors.cpu(electric_field_mps))
    psi_tt = TCI.TensorTrain(ITensors.cpu(psi_mps))

    R = phase.R
    localdims = fill(2, 2 * phase.R)

    function kernel(q_bits::AbstractVector{Int})
        q_x, q_kv = split_interleaved_bits(q_bits, R)
        q_kv_aligned = maybe_reverse_bits(q_kv, phase.kv_lsb_first)

        kv_orig = quantics_to_origcoord(phase.kv_grid, q_kv_aligned)
        n_v = k_to_n(kv_orig, phase.M)
        kv_phys = 2π * n_v / phase.Lv

        E_val = electric_field_tt(q_x)

        phase_angle = E_val * kv_phys * params.dt
        return exp(im * phase_angle * frequency_filter(n_v; beta=params.beta, k_cut=params.k_cut)) * psi_tt(q_bits) 
    end
    #kernel = TCI.ThreadedBatchEvaluator{ComplexF64}(kernel, localdims)
    kernel = TCI.CachedFunction{ComplexF64}(kernel, localdims)

    # if previous_tci === nothing
    #     pivots = acceleration_pivots(phase.x_grid, phase.kv_grid, phase.M; lsb_first=true)
    # else
    #     pivots = collect_final_pivots(previous_tci)
    #     println(pivots)
    # end

    # psi_tt = TCI.crossinterpolate2(
    #     ComplexF64,
    #     kernel,
    #     fill(2, 2*phase.R),
    #     pivots;
    #     tolerance = params.tolerance,
    #     maxbonddim = params.maxrank,
    #     verbosity = 1
    # )[1]

    tci_start = time()
    if previous_tci === nothing
        pivots = acceleration_pivots(
            phase.x_grid,
            phase.kv_grid,
            phase.M;
            x_lsb_first = phase.x_lsb_first,
            kv_lsb_first = phase.kv_lsb_first,
        )
        pivots = TCI.optfirstpivot(kernel, localdims)

        psi_tt = TCI.crossinterpolate2(
            ComplexF64,
            kernel,
            localdims,
            [pivots];
            tolerance = params.tolerance,
            maxbonddim = params.maxrank,
        )[1]
    else
        TCI.optimize!(
            previous_tci, 
            kernel; 
            tolerance = params.tolerance, 
            maxbonddim = params.maxrank, 
            verbosity = 0,
            )
        psi_tt = previous_tci
    end
    tci_time = time() - tci_start
    if tci_profile
        label = isempty(tci_profile_label) ? "" : " " * tci_profile_label
        println("TCI fit stats$(label): tci_time=$(round(tci_time, digits=3)) s")
        println("--")
    end

    psi_mps = MPS(psi_tt; sites = mps_sites)
    if params.use_gpu
        psi_mps = cu(psi_mps)
    end
    
    mpo_start = time()
    # Inverse Fourier transform in v, Fourier transform in x
    psi_mps = apply(v_inv_x_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Free streaming step
    psi_mps = apply(free_stream_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)
    
    # Inverse Fourier transform in x, Fourier transform in v
    psi_mps = apply(x_inv_v_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)
    mpo_time = time() - mpo_start
    step_time = time() - steptime_start

    return psi_mps, electric_field_mps, psi_tt, (tci_time, mpo_time, step_time)
end

function strang_step_filtered_RK4!(
    psi_mps::MPS,
    phase::PhaseSpaceGrids,
    full_poisson_mpo::TCI.TensorTrain,
    free_stream_mpo_it::MPO,
    x_fourier_mpo_it::MPO,
    x_inv_fourier_mpo_it::MPO,
    v_fourier_mpo_it::MPO,
    v_inv_fourier_mpo_it::MPO,
    stretched_kv_mpo_it::MPO,
    half_filtering_mpo_it::MPO,
    mpo_sites;
    params::SimulationParams,
    return_field::Bool = false,
)

    # Get electric field MPO
    _, electric_field_mps = get_electric_field_mps_kv(
        psi_mps,
        full_poisson_mpo;
        dv = sqrt(phase.M) * phase.dv,
        cutoff = params.cutoff,
        maxrank_ef = params.maxrank_ef,
        alg = params.alg,
    )

    electric_field_tt = TCI.TensorTrain(ITensors.cpu(electric_field_mps))
    electric_field_mpo = tt_to_mpo(electric_field_tt)
    stretched_electric_field_mpo_it = MPO(stretched_mpo(electric_field_mpo, 1, 2); sites = mpo_sites)

    if params.use_gpu
        stretched_electric_field_mpo_it = cu(stretched_electric_field_mpo_it)
    end

    # RK4 update
    B = 1im * apply(stretched_electric_field_mpo_it, stretched_kv_mpo_it; alg = "naive")
    dt = params.dt

    k1 = apply(B, psi_mps; alg = "naive")

    psi_tmp = add(psi_mps, (dt/2) * k1; maxdim = params.maxrank, cutoff = params.cutoff)
    k2 = apply(B, psi_tmp; alg = "naive")

    psi_tmp = add(psi_mps, (dt/2) * k2; maxdim = params.maxrank, cutoff = params.cutoff)
    k3 = apply(B, psi_tmp; alg = "naive")

    psi_tmp = add(psi_mps, dt * k3; maxdim = params.maxrank, cutoff = params.cutoff)
    k4 = apply(B, psi_tmp; alg = "naive")

    psi_mps = add(psi_mps, (dt / 6) * (k1 + 2k2 + 2k3 + k4); maxdim = params.maxrank, cutoff = params.cutoff)
    
    # Fourier transform in x
    psi_mps = apply(x_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Filtering 
    psi_mps = apply(half_filtering_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Inverse Fourier transform in v
    psi_mps = apply(v_inv_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Free streaming step
    psi_mps = apply(free_stream_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Fourier transform in v
    psi_mps = apply(v_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Filtering
    psi_mps = apply(half_filtering_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Inverse Fourier transform in x
    psi_mps = apply(x_inv_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    return return_field ? (psi_mps, electric_field_mps) : psi_mps
end

function strang_step_unfiltered_RK4!(
    psi_mps::MPS,
    phase::PhaseSpaceGrids,
    full_poisson_mpo::TCI.TensorTrain,
    free_stream_mpo_it::MPO,
    x_fourier_mpo_it::MPO,
    x_inv_fourier_mpo_it::MPO,
    v_fourier_mpo_it::MPO,
    v_inv_fourier_mpo_it::MPO,
    x_inv_v_fourier_mpo_it::MPO,
    v_inv_x_fourier_mpo_it::MPO,
    stretched_kv_mpo_it::MPO,
    mpo_sites;
    params::SimulationParams,
    return_field::Bool = false,
)

    # Get electric field MPO
    _, electric_field_mps = get_electric_field_mps_kv(
        psi_mps,
        full_poisson_mpo;
        dv = sqrt(phase.M) * phase.dv,
        cutoff = params.cutoff,
        maxrank_ef = params.maxrank_ef,
        alg = params.alg,
    )

    electric_field_tt = TCI.TensorTrain(ITensors.cpu(electric_field_mps))
    electric_field_mpo = tt_to_mpo(electric_field_tt)
    stretched_electric_field_mpo_it = MPO(stretched_mpo(electric_field_mpo, 1, 2); sites = mpo_sites)

    if params.use_gpu
        stretched_electric_field_mpo_it = cu(stretched_electric_field_mpo_it)
    end

    # RK4 update
    B = 1im * apply(stretched_electric_field_mpo_it, stretched_kv_mpo_it; alg = "naive")
    dt = params.dt

    k1 = apply(B, psi_mps; alg = "naive")

    psi_tmp = add(psi_mps, (dt/2) * k1; maxdim = params.maxrank, cutoff = params.cutoff)
    k2 = apply(B, psi_tmp; alg = "naive")

    psi_tmp = add(psi_mps, (dt/2) * k2; maxdim = params.maxrank, cutoff = params.cutoff)
    k3 = apply(B, psi_tmp; alg = "naive")

    psi_tmp = add(psi_mps, dt * k3; maxdim = params.maxrank, cutoff = params.cutoff)
    k4 = apply(B, psi_tmp; alg = "naive")

    psi_mps = add(psi_mps, (dt / 6) * (k1 + 2k2 + 2k3 + k4); maxdim = params.maxrank, cutoff = params.cutoff)

    # Inverse Fourier transform in v, Fourier transform in x
    psi_mps = apply(v_inv_x_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Free streaming step
    psi_mps = apply(free_stream_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Inverse Fourier transform in x, Fourier transform in v
    psi_mps = apply(x_inv_v_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    return return_field ? (psi_mps, electric_field_mps) : psi_mps
end
