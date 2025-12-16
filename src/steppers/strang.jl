Base.@kwdef struct SimulationParams
    dt::Float64
    tolerance::Float64 = 1e-8
    cutoff::Float64 = 1e-8
    maxrank::Union{Int,Nothing} = nothing
    maxrank_ef::Union{Int,Nothing} = nothing
    k_cut::Real = 2^6
    beta::Real = 2.0
    v0::Real = 1e-1
    use_gpu::Bool = false
    alg::String = "naive"
    l1_normalize::Bool = true
end

const strang_pivots_cache = Ref{Union{Nothing,Vector{Vector{Int}}}}(nothing)

function strang_step_v2!(
    psi_mps::MPS,
    phase::PhaseSpaceGrids,
    full_poisson_mpo::TCI.TensorTrain,
    free_stream_mpo_it::MPO,
    sites_mps;
    params::SimulationParams,
    target_norm::Union{Real,Nothing}=nothing,
    return_field::Bool=false,
)

    # Get electric field TT
    electric_field_mps = get_electric_field_mps(
        psi_mps,
        full_poisson_mpo;
        dv = phase.dv,
        cutoff = params.cutoff,
        maxrank_ef = params.maxrank,
        alg = params.alg,
    )
    # TCI conversion only supports CPU arrays, so copy off GPU if needed
    electric_field_tt = TCI.TensorTrain(ITensors.cpu(electric_field_mps))
    psi_tt_old = TCI.TensorTrain(ITensors.cpu(psi_mps))

    function evolved_psi(q::Vector{Int})
        # Split quantics bits
        qx_bits = q[1:2:2*phase.R]
        qv_bits = q[2:2:2*phase.R]

        # Decode physical coordinates
        x = quantics_to_origcoord(phase.x_grid, qx_bits)
        v = quantics_to_origcoord(phase.v_grid, qv_bits)

        # Evaluate electric field at x (using same quantics encoding)
        # Use only the real part to keep shifted coordinates real-valued
        E_x = real(electric_field_tt(qx_bits))

        # Compute shifted velocity
        v_shifted = v + params.dt * sin(3 * 2*pi/phase.Lx * x) * 0.1 #E_x
        #v_shifted = clamp(v_shifted, phase.vmin, phase.vmax)
        # Period boundary conditions in v
        v_range = phase.vmax - phase.vmin
        if v_shifted < phase.vmin
            v_shifted += v_range * ceil((phase.vmin - v_shifted) / v_range)
        elseif v_shifted >= phase.vmax
            v_shifted -= v_range * ceil((v_shifted - phase.vmax) / v_range)
        end

        # Convert shifted v back to quantics bits
        qv_bits_shifted = origcoord_to_quantics(phase.v_grid, v_shifted)

        # Combine coordinates into a single bit vector
        q_shifted = interleave_bits(qx_bits, qv_bits_shifted)

        # Return the old state evaluated at the shifted point
        return psi_tt_old(q_shifted)
    end

    # Perform a TCI fit of psi_mps <- psi_mps(x, v + dt E)
    localdims = fill(2, 2*phase.R)
    cached_evolved_psi = TCI.CachedFunction{Float64}(evolved_psi, localdims)

    println("Applying acceleration step via TCI...")

    cached_pivots = strang_pivots_cache[]
    pivots = (cached_pivots === nothing || any(length(p) != length(localdims) for p in cached_pivots)) ?
        nothing : cached_pivots

    function run_crossinterpolate(pivots_arg)
        if pivots_arg === nothing
            return TCI.crossinterpolate2(
                Float64, cached_evolved_psi, localdims;
                tolerance = params.tolerance, maxbonddim = params.maxrank,
            )[1]
        end
        return TCI.crossinterpolate2(
            Float64, cached_evolved_psi, localdims, pivots_arg;
            tolerance = params.tolerance, maxbonddim = params.maxrank,
        )[1]
    end

    psi_tci_new = try
        run_crossinterpolate(pivots)
    catch err
        if err isa DimensionMismatch
            @warn "Cached TCI pivots invalid for current layout; recomputing from scratch" err
            strang_pivots_cache[] = nothing
            run_crossinterpolate(nothing)
        else
            rethrow(err)
        end
    end

    strang_pivots_cache[] = collect_final_pivots(psi_tci_new)
    
    #  Pivot at origin 
    # pivot = [interleave_bits(
    #     origcoord_to_quantics(phase.x_grid, 0.0),
    #     origcoord_to_quantics(phase.v_grid, 0.0),
    # )]
    # psi_tci_new = TCI.crossinterpolate2(Float64, evolved_psi, localdims; tolerance = params.tolerance, maxbonddim = params.maxrank)[1]

    println("Acceleration MPO ranks: ", TCI.rank(psi_tci_new))

    psi_tt_new = TCI.TensorTrain(psi_tci_new)
    psi_mps = MPS(psi_tt_new; sites = sites_mps)
    if params.use_gpu
        psi_mps = cu(psi_mps)
    end

    # Apply free streaming MPO
    println("Applying free streaming step via MPO...")
    # psi_mps = apply(free_stream_mpo_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)

    if target_norm !== nothing
        current_l1 = total_charge(psi_mps, phase)
        psi_mps .= (target_norm / current_l1) * psi_mps
    end

    return return_field ? (psi_mps, electric_field_mps) : psi_mps

    # accel_mpo_half = get_acceleration_mpo(
    #     params.dt / 2,
    #     phase.Lv,
    #     phase.M,
    #     phase.x_grid,
    #     phase.kv_grid,
    #     TCI.TensorTrain(electric_field_mps);
    #     tolerance = params.tolerance,
    #     lsb_first = true,
    #     k_cut = params.k_cut,
    #     beta = params.beta,
    #     accel_cache = accel_cache,
    #     reuse_strategy = reuse_strategy,
    # )
    # accel_mpo_half_it = MPO(accel_mpo_half; sites = sites_mpo)
    # if params.use_gpu
    #     accel_mpo_half_it = cu(accel_mpo_half_it)
    # end

    # psi_mps = apply(v_fourier_mpo_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)
    # psi_mps = apply(accel_mpo_half_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)
    # psi_mps = apply(v_inv_fourier_mpo_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)

    # psi_mps = apply(free_stream_mpo_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)

    # electric_field_mps = get_electric_field_mps(
    #     psi_mps,
    #     full_poisson_mpo;
    #     dv = phase.dv,
    #     tolerance = params.tolerance,
    #     maxrank = params.maxrank,
    #     alg = params.alg,
    # )

    # accel_mpo_half = get_acceleration_mpo(
    #     params.dt / 2,
    #     phase.Lv,
    #     phase.M,
    #     phase.x_grid,
    #     phase.kv_grid,
    #     TCI.TensorTrain(electric_field_mps);
    #     tolerance = params.tolerance,
    #     lsb_first = true,
    #     k_cut = params.k_cut,
    #     beta = params.beta,
    #     accel_cache = accel_cache,
    #     reuse_strategy = reuse_strategy,
    # )
    # accel_mpo_half_it = MPO(accel_mpo_half; sites = sites_mpo)
    # if params.use_gpu
    #     accel_mpo_half_it = cu(accel_mpo_half_it)
    # end

    # psi_mps = apply(v_fourier_mpo_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)
    # psi_mps = apply(accel_mpo_half_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)
    # psi_mps = apply(v_inv_fourier_mpo_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)
end

function strang_step_v3!(
    psi_mps::MPS,
    phase::PhaseSpaceGrids,
    full_poisson_mpo::TCI.TensorTrain,
    free_stream_mpo_it::MPO,
    v_fourier_mpo_it::MPO,
    v_inv_fourier_mpo_it::MPO,
    stretched_kv_mpo_it::MPO,
    diffusion_mpo_it::MPO,
    sites_mpo;
    params::SimulationParams,
    target_norm::Union{Real,Nothing}=nothing,
    return_field::Bool=false,
)

    # Get electric field MPO
    _, electric_field_mps = get_electric_field_mps(
        psi_mps,
        full_poisson_mpo;
        dv = phase.dv,
        cutoff = params.cutoff,
        maxrank_ef = params.maxrank,
        alg = params.alg,
    )
    # TODO: Implement MPS -> MPO conversion in ITensor
    electric_field_tt = TCI.TensorTrain(ITensors.cpu(electric_field_mps))
    electric_field_mpo = tt_to_mpo(electric_field_tt)
    stretched_electric_field_mpo_it = MPO(stretched_mpo(electric_field_mpo, 1, 2); sites = sites_mpo)

    if params.use_gpu
        stretched_electric_field_mpo_it = cu(stretched_electric_field_mpo_it)
    end

    # Fourier transform in v
    psi_hat = apply(v_fourier_mpo_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.cutoff)

    psi_hat = apply(diffusion_mpo_it, psi_hat; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.cutoff)

    B = 1im * apply(stretched_electric_field_mpo_it, stretched_kv_mpo_it; alg = "naive")
    dt = params.dt

    # RK4 update
    k1 = apply(B, psi_hat; alg = "naive")
    #psi_tmp = psi_hat + (dt/2) * k1
    #psi_tmp = truncate!(psi_hat + (dt/2) * k1; maxdim = params.maxrank, cutoff = params.cutoff)
    psi_tmp = add(psi_hat, (dt/2) * k1; maxdim = params.maxrank, cutoff = params.cutoff)

    k2 = apply(B, psi_tmp; alg = "naive")
    #psi_tmp = psi_hat + (dt/2) * k2
    #psi_tmp = truncate!(psi_hat + (dt/2) * k2; maxdim = params.maxrank, cutoff = params.cutoff)
    psi_tmp = add(psi_hat, (dt/2) * k2; maxdim = params.maxrank, cutoff = params.cutoff)

    k3 = apply(B, psi_tmp; alg = "naive")
    #psi_tmp = psi_hat + dt * k3
    #psi_tmp = truncate!(psi_hat + dt * k3; maxdim = params.maxrank, cutoff = params.cutoff)
    psi_tmp = add(psi_hat, dt * k3; maxdim = params.maxrank, cutoff = params.cutoff)

    k4 = apply(B, psi_tmp; alg = "naive")

    psi_hat_evolved = truncate!(psi_hat + (dt / 6) * (k1 + 2k2 + 2k3 + k4); maxdim = params.maxrank, cutoff = params.cutoff)
    
    # Inverse Fourier transform in v
    psi_mps = apply(v_inv_fourier_mpo_it, psi_hat_evolved; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.cutoff)

    # Free streaming step
    psi_mps = apply(free_stream_mpo_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.cutoff)

    # Normalize according to L1 norm
    if target_norm !== nothing && params.l1_normalize == true
        current_charge = total_charge(psi_mps, phase)
        psi_mps .= (target_norm / current_charge) * psi_mps
    end

    return return_field ? (psi_mps, electric_field_mps) : psi_mps
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
    mps_sites;
    params::SimulationParams,
    target_norm::Union{Real,Nothing}=nothing,
    return_field::Bool=false,
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

    function kernel(q_bits::AbstractVector{Int})
        q_x = q_bits[1:2:2*phase.R]
        q_kv = reverse(q_bits[2:2:2*phase.R])

        kv_orig = quantics_to_origcoord(phase.kv_grid, q_kv)
        n_v = k_to_n(kv_orig, phase.M)
        kv_phys = 2π * n_v / phase.Lv

        E_val = electric_field_tt(q_x)

        phase_angle = E_val * kv_phys * params.dt
        return exp(im * phase_angle) * psi_tt(q_bits) * frequency_filter(kv_phys; beta=params.beta, k_cut=params.k_cut)
    end

    psi_tt = TCI.crossinterpolate2(
        ComplexF64,
        kernel,
        fill(2, 2*phase.R),
        acceleration_pivots(phase.x_grid, phase.kv_grid, phase.M; lsb_first=true);
        tolerance = params.tolerance,
        maxbonddim = params.maxrank,
    )[1]

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

    if target_norm !== nothing && params.l1_normalize == true
        current_charge = total_charge(psi_mps, phase)
        psi_mps .= (target_norm / current_charge) * psi_mps
    end

    return return_field ? (psi_mps, electric_field_mps) : psi_mps
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
    mps_sites;
    params::SimulationParams,
    target_norm::Union{Real,Nothing}=nothing,
    return_field::Bool=false,
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

    function kernel(q_bits::AbstractVector{Int})
        q_x = q_bits[1:2:2*phase.R]
        q_kv = reverse(q_bits[2:2:2*phase.R])

        kv_orig = quantics_to_origcoord(phase.kv_grid, q_kv)
        n_v = k_to_n(kv_orig, phase.M)
        kv_phys = 2π * n_v / phase.Lv

        E_val = electric_field_tt(q_x)

        phase_angle = E_val * kv_phys * params.dt
        return exp(im * phase_angle) * psi_tt(q_bits) * frequency_filter(kv_phys; beta=params.beta, k_cut=params.k_cut)
    end

    psi_tt = TCI.crossinterpolate2(
        ComplexF64,
        kernel,
        fill(2, 2*phase.R),
        acceleration_pivots(phase.x_grid, phase.kv_grid, phase.M; lsb_first=true);
        tolerance = params.tolerance,
        maxbonddim = params.maxrank,
    )[1]

    psi_mps = MPS(psi_tt; sites = mps_sites)
    if params.use_gpu
        psi_mps = cu(psi_mps)
    end

    # Inverse Fourier transform in v
    psi_mps = apply(v_inv_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Fourier transform in x
    psi_mps = apply(x_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Free streaming step
    psi_mps = apply(free_stream_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Inverse Fourier transform in x
    psi_mps = apply(x_inv_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Fourier transform in v
    psi_mps = apply(v_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    if target_norm !== nothing && params.l1_normalize == true
        current_charge = total_charge(psi_mps, phase)
        psi_mps .= (target_norm / current_charge) * psi_mps
    end

    return return_field ? (psi_mps, electric_field_mps) : psi_mps
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
    target_norm::Union{Real,Nothing}=nothing,
    return_field::Bool=false,
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

    if target_norm !== nothing && params.l1_normalize == true
        current_charge = total_charge(psi_mps, phase)
        psi_mps .= (target_norm / current_charge) * psi_mps
    end

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
    stretched_kv_mpo_it::MPO,
    mpo_sites;
    params::SimulationParams,
    target_norm::Union{Real,Nothing}=nothing,
    return_field::Bool=false,
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

    # Inverse Fourier transform in v
    psi_mps = apply(v_inv_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Fourier transform in x
    psi_mps = apply(x_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Free streaming step
    psi_mps = apply(free_stream_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Inverse Fourier transform in x
    psi_mps = apply(x_inv_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    # Fourier transform in v
    psi_mps = apply(v_fourier_mpo_it, psi_mps; alg = params.alg, maxdim = params.maxrank, cutoff = params.cutoff)

    if target_norm !== nothing && params.l1_normalize == true
        current_charge = total_charge(psi_mps, phase)
        psi_mps .= (target_norm / current_charge) * psi_mps
    end

    return return_field ? (psi_mps, electric_field_mps) : psi_mps
end

# Keep the original exported name pointing to the updated implementation.
const strang_step! = strang_step_v2!
