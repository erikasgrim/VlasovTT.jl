Base.@kwdef struct SimulationParams
    dt::Float64
    tolerance::Float64 = 1e-8
    maxrank::Union{Int,Nothing} = nothing
    k_cut::Real = 2^6
    beta::Real = 2.0
    use_gpu::Bool = false
    alg::String = "naive"
end

function strang_step!(
    psi_mps::MPS,
    phase::PhaseSpaceGrids,
    full_poisson_mpo::TCI.TensorTrain,
    free_stream_mpo_it::MPO,
    v_fourier_mpo_it::MPO,
    v_inv_fourier_mpo_it::MPO,
    sites_mpo;
    params::SimulationParams,
    target_norm::Union{Real,Nothing}=nothing,
    accel_cache::Union{AccelerationTCICache,Nothing}=nothing,
    return_field::Bool=false,
    reuse_strategy::Symbol=:resweep,
)
    electric_field_mps = get_electric_field_mps(
        psi_mps,
        full_poisson_mpo;
        dv = phase.dv,
        tolerance = params.tolerance,
        maxrank = params.maxrank,
        alg = params.alg,
    )

    accel_mpo_half = get_acceleration_mpo(
        params.dt / 2,
        phase.Lv,
        phase.M,
        phase.x_grid,
        phase.kv_grid,
        TCI.TensorTrain(electric_field_mps);
        tolerance = params.tolerance,
        lsb_first = true,
        k_cut = params.k_cut,
        beta = params.beta,
        accel_cache = accel_cache,
        reuse_strategy = reuse_strategy,
    )
    accel_mpo_half_it = MPO(accel_mpo_half; sites = sites_mpo)
    if params.use_gpu
        accel_mpo_half_it = cu(accel_mpo_half_it)
    end

    psi_mps = apply(v_fourier_mpo_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)
    psi_mps = apply(accel_mpo_half_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)
    psi_mps = apply(v_inv_fourier_mpo_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)

    psi_mps = apply(free_stream_mpo_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)

    electric_field_mps = get_electric_field_mps(
        psi_mps,
        full_poisson_mpo;
        dv = phase.dv,
        tolerance = params.tolerance,
        maxrank = params.maxrank,
        alg = params.alg,
    )

    accel_mpo_half = get_acceleration_mpo(
        params.dt / 2,
        phase.Lv,
        phase.M,
        phase.x_grid,
        phase.kv_grid,
        TCI.TensorTrain(electric_field_mps);
        tolerance = params.tolerance,
        lsb_first = true,
        k_cut = params.k_cut,
        beta = params.beta,
        accel_cache = accel_cache,
        reuse_strategy = reuse_strategy,
    )
    accel_mpo_half_it = MPO(accel_mpo_half; sites = sites_mpo)
    if params.use_gpu
        accel_mpo_half_it = cu(accel_mpo_half_it)
    end

    psi_mps = apply(v_fourier_mpo_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)
    psi_mps = apply(accel_mpo_half_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)
    psi_mps = apply(v_inv_fourier_mpo_it, psi_mps; alg = params.alg, truncate = true, maxdim = params.maxrank, cutoff = params.tolerance)

    if target_norm !== nothing
        current_l1 = total_charge(psi_mps, phase)
        psi_mps .= (target_norm / current_l1) * psi_mps
    end

    return return_field ? (psi_mps, electric_field_mps) : psi_mps
end
