struct SolverMPOs
    x_fourier_mpo::TCI.TensorTrain
    v_fourier_mpo::TCI.TensorTrain
    x_inv_fourier_mpo::TCI.TensorTrain
    v_inv_fourier_mpo::TCI.TensorTrain
    fourier_mpo::TCI.TensorTrain
    inv_fourier_mpo::TCI.TensorTrain
    free_streaming_mpo::TCI.TensorTrain
    full_free_streaming_mpo::TCI.TensorTrain
    poisson_mpo::TCI.TensorTrain
    full_poisson_mpo::TCI.TensorTrain
end

function build_fourier_mpos(R::Int; tolerance::Real = 1e-8, lsb_first::Bool = false)
    x_fourier_mpo = stretched_fourier_mpo(R, 1, 2; sign = -1.0, tolerance = tolerance)
    v_fourier_mpo = stretched_fourier_mpo(R, 2, 2; sign = -1.0, tolerance = tolerance)
    x_inv_fourier_mpo = stretched_fourier_mpo(R, 1, 2; sign = 1.0, tolerance = tolerance, lsb_first = true)
    v_inv_fourier_mpo = stretched_fourier_mpo(R, 2, 2; sign = 1.0, tolerance = tolerance, lsb_first = true)

    fourier_mpo = quanticsfouriermpo(R; sign = -1.0, tolerance = tolerance)
    inv_fourier_mpo = quanticsfouriermpo(R; sign = 1.0, tolerance = tolerance)

    return (
        x_fourier_mpo = x_fourier_mpo,
        v_fourier_mpo = v_fourier_mpo,
        x_inv_fourier_mpo = x_inv_fourier_mpo,
        v_inv_fourier_mpo = v_inv_fourier_mpo,
        fourier_mpo = fourier_mpo,
        inv_fourier_mpo = inv_fourier_mpo,
    )
end

function build_solver_mpos(
    phase::PhaseSpaceGrids;
    dt::Real,
    tolerance::Real = 1e-8,
    k_cut::Real = 2^6,
    beta::Real = 2.0,
    eps0::Real = 1.0,
    lsb_first::Bool = true,
)
    fourier = build_fourier_mpos(phase.R; tolerance = tolerance, lsb_first = lsb_first)

    free_streaming_mpo = get_free_streaming_mpo(
        dt,
        phase.Lx,
        phase.M,
        phase.kx_grid,
        phase.v_grid;
        tolerance = tolerance,
        k_cut = k_cut,
        beta = beta,
        lsb_first = lsb_first,
    )

    full_free_streaming_mpo = TCI.contract(
        fourier.x_inv_fourier_mpo,
        TCI.contract(
            free_streaming_mpo,
            fourier.x_fourier_mpo;
            algorithm = :naive,
            tolerance = tolerance,
        );
        algorithm = :naive,
        tolerance = tolerance,
    )

    poisson_mpo = get_poisson_mpo(
        phase.Lx,
        phase.M,
        phase.kx_grid;
        tolerance = tolerance,
        eps0 = eps0,
    )

    full_poisson_mpo = TCI.contract(
        reverse(fourier.inv_fourier_mpo),
        TCI.contract(
            reverse(poisson_mpo),
            fourier.fourier_mpo;
            algorithm = :naive,
            tolerance = tolerance,
        );
        algorithm = :naive,
        tolerance = tolerance,
    )

    return SolverMPOs(
        fourier.x_fourier_mpo,
        fourier.v_fourier_mpo,
        fourier.x_inv_fourier_mpo,
        fourier.v_inv_fourier_mpo,
        fourier.fourier_mpo,
        fourier.inv_fourier_mpo,
        free_streaming_mpo,
        full_free_streaming_mpo,
        poisson_mpo,
        full_poisson_mpo,
    )
end

function prepare_itensor_mpos(
    mpos::SolverMPOs,
    sites_mpo;
    use_gpu::Bool = false,
)
    free_stream_mpo_it = MPO(mpos.full_free_streaming_mpo; sites = sites_mpo)
    v_fourier_mpo_it = MPO(mpos.v_fourier_mpo; sites = sites_mpo)
    v_inv_fourier_mpo_it = MPO(mpos.v_inv_fourier_mpo; sites = sites_mpo)

    if use_gpu
        free_stream_mpo_it = cu(free_stream_mpo_it)
        v_fourier_mpo_it = cu(v_fourier_mpo_it)
        v_inv_fourier_mpo_it = cu(v_inv_fourier_mpo_it)
    end

    return (
        free_stream = free_stream_mpo_it,
        v_fourier = v_fourier_mpo_it,
        v_inv_fourier = v_inv_fourier_mpo_it,
    )
end
