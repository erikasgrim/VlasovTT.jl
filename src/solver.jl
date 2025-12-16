struct SolverMPOs
    x_fourier_mpo::TCI.TensorTrain
    v_fourier_mpo::TCI.TensorTrain
    x_inv_fourier_mpo::TCI.TensorTrain
    v_inv_fourier_mpo::TCI.TensorTrain
    fourier_mpo::TCI.TensorTrain
    inv_fourier_mpo::TCI.TensorTrain
    half_free_streaming_fourier_mpo::TCI.TensorTrain
    full_free_streaming_fourier_mpo::TCI.TensorTrain
    poisson_mpo::TCI.TensorTrain
    full_poisson_mpo::TCI.TensorTrain
    stretched_kv_mpo::TCI.TensorTrain
    diffusion_mpo::TCI.TensorTrain
    half_filtering_mpo::TCI.TensorTrain
end

function build_fourier_mpos(R::Int; tolerance::Real = 1e-8, lsb_first::Bool = false)
    fourier_tol = 1e-12
    x_fourier_mpo = stretched_fourier_mpo(R, 1, 2; sign = -1.0, tolerance = fourier_tol)
    v_fourier_mpo = stretched_fourier_mpo(R, 2, 2; sign = -1.0, tolerance = fourier_tol)
    x_inv_fourier_mpo = stretched_fourier_mpo(R, 1, 2; sign = 1.0, tolerance = fourier_tol, lsb_first = true)
    v_inv_fourier_mpo = stretched_fourier_mpo(R, 2, 2; sign = 1.0, tolerance = fourier_tol, lsb_first = true)

    fourier_mpo = quanticsfouriermpo(R; sign = -1.0, tolerance = fourier_tol)
    inv_fourier_mpo = quanticsfouriermpo(R; sign = 1.0, tolerance = fourier_tol)

    return (
        x_fourier_mpo = x_fourier_mpo,
        v_fourier_mpo = v_fourier_mpo,
        x_inv_fourier_mpo = x_inv_fourier_mpo,
        v_inv_fourier_mpo = v_inv_fourier_mpo,
        fourier_mpo = fourier_mpo,
        inv_fourier_mpo = inv_fourier_mpo,
    )
end

# function build_kv_mpo(Lv, M, kv_grid; tolerance::Real = 1e-12, k_cut::Real = 2^8, beta::Real = 2.0)
#     function kv_kernel(kv)
#         q_bits_normal = origcoord_to_quantics(kv_grid, kv)
#         q_bits_reversed = reverse(q_bits_normal)
#         kv_reversed = quantics_to_origcoord(kv_grid, q_bits_reversed)

#         nv = k_to_n(kv_reversed, M)

#         return 2pi * nv / Lv * frequency_filter(nv; k_cut = k_cut, beta = beta)
#     end

#     initial_pivots = [origcoord_to_quantics(kv_grid, n_to_k(n, M)) for n in -10:10]

#     qtci, _, _ = quanticscrossinterpolate(
#         Float64,
#         kv_kernel,
#         kv_grid,
#         initial_pivots;
#         tolerance = tolerance,
#     )
#     return tt_to_mpo(TCI.TensorTrain(qtci.tci))
# end

function build_kv_mpo(Lv, M, kv_grid; tolerance::Real = 1e-12, k_cut::Real = 2^8, beta::Real = 2.0)
    function kv_kernel(q_bits)
        #q_bits_normal = origcoord_to_quantics(kv_grid, kv)
        #q_bits_reversed = reverse(q_bits_normal)
        kv_reversed = quantics_to_origcoord(kv_grid, reverse(q_bits))

        nv = k_to_n(kv_reversed, M)

        return 2pi * nv / Lv * frequency_filter(nv; k_cut = k_cut, beta = beta)
    end

    initial_pivots = [origcoord_to_quantics(kv_grid, n_to_k(n, M)) for n in -10:10]
    R = length(kv_grid)
    localdims = fill(2, R)

    tci, _, _ = TCI.crossinterpolate2(
        Float64,
        kv_kernel,
        localdims,
        initial_pivots;
        tolerance = tolerance,
    )
    return tt_to_mpo(TCI.TensorTrain(tci))
end

function build_diffusion_mpo(dt, Lv, M, kv_grid; tolerance::Real = 1e-12, nu::Real = 3e-5, k_cut::Real = 2^8, beta::Real = 2.0)
    function kv_kernel(q_bits)
        #q_bits_normal = origcoord_to_quantics(kv_grid, kv)
        #q_bits_reversed = reverse(q_bits_normal)
        kv = quantics_to_origcoord(kv_grid, reverse(q_bits))

        nv = k_to_n(kv, M)

        return exp(- nu * (2pi * nv / Lv)^2 * dt)# * frequency_filter(nv; k_cut = k_cut, beta = beta)
    end

    initial_pivots = [origcoord_to_quantics(kv_grid, n_to_k(n, M)) for n in -10:10]
    R = length(kv_grid)
    localdims = fill(2, R)

    tci, _, _ = TCI.crossinterpolate2(
        Float64,
        kv_kernel,
        localdims,
        initial_pivots;
        tolerance = tolerance,
    )
    println("Diffusion MPO ranks: ", TCI.rank(tci))
    return tt_to_mpo(TCI.TensorTrain(tci))
end

function build_filtering_mpo(dt, Lv, Lx, M, kv_grid, kx_grid; tolerance::Real = 1e-12, v0::Real = 1e-1)
    function filter_kernel(q_bits)
        q_kx = q_bits[1:2:end]
        q_kv = q_bits[2:2:end]
        kx = quantics_to_origcoord(kx_grid, reverse(q_kx))
        kv = quantics_to_origcoord(kv_grid, reverse(q_kv))

        n_x = k_to_n(kx, M)
        n_v = k_to_n(kv, M)
        
        return exp(v0^2 * (2pi / Lv) * n_v * (2pi / Lx) * n_x * dt)
    end

    initial_pivots_x = [origcoord_to_quantics(kx_grid, n_to_k(n, M)) for n in -5:5]
    initial_pivots_v = [origcoord_to_quantics(kv_grid, n_to_k(n, M)) for n in -5:5]
    initial_pivots = interleave_bits(initial_pivots_x, initial_pivots_v)
    R = length(kx_grid)
    localdims = fill(2, 2R)

    tci, _, _ = TCI.crossinterpolate2(
        Float64,
        filter_kernel,
        localdims,
        initial_pivots;
        tolerance = tolerance,
    )
    println("Filamentation filter MPO ranks: ", TCI.rank(tci))
    return tt_to_mpo(TCI.TensorTrain(tci))
end

function build_solver_mpos(
    phase::PhaseSpaceGrids;
    dt::Real,
    tolerance::Real = 1e-8,
    k_cut::Real = 2^8,
    beta::Real = 2.0,
    eps0::Real = 1.0,
    v0::Real = 1e-1,
    lsb_first::Bool = true,
)
    fourier = build_fourier_mpos(phase.R; tolerance = tolerance, lsb_first = lsb_first)

    full_free_streaming_fourier_mpo = get_free_streaming_mpo(
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

    half_free_streaming_fourier_mpo = get_free_streaming_mpo(
        dt / 2,
        phase.Lx,
        phase.M,
        phase.kx_grid,
        phase.v_grid;
        tolerance = tolerance,
        k_cut = k_cut,
        beta = beta,
        lsb_first = lsb_first,
    )

    diffusion_mpo = build_diffusion_mpo(
        dt,
        phase.Lv,
        phase.M,
        phase.kv_grid;
        tolerance = tolerance,
        k_cut = k_cut,
        beta = beta,
    )

    diffusion_mpo = stretched_mpo(diffusion_mpo, 2, 2)

    half_filtering_mpo = build_filtering_mpo(
        dt / 2,
        phase.Lv,
        phase.Lx,
        phase.M,
        phase.kv_grid,
        phase.kx_grid;
        tolerance = tolerance,
        v0 = v0,
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

    kv_mpo = build_kv_mpo(
        phase.Lv, 
        phase.M, 
        phase.kv_grid; 
        tolerance = tolerance,
        k_cut = k_cut,
        beta = beta,
    )

    stretched_kv_mpo = stretched_mpo(kv_mpo, 2, 2)

    return SolverMPOs(
        fourier.x_fourier_mpo,
        fourier.v_fourier_mpo,
        fourier.x_inv_fourier_mpo,
        fourier.v_inv_fourier_mpo,
        fourier.fourier_mpo,
        fourier.inv_fourier_mpo,
        half_free_streaming_fourier_mpo,
        full_free_streaming_fourier_mpo,
        poisson_mpo,
        full_poisson_mpo,
        stretched_kv_mpo,
        diffusion_mpo,
        half_filtering_mpo,
    )
end

function prepare_itensor_mpos(
    mpos::SolverMPOs,
    sites_mpo;
    use_gpu::Bool = false,
    cutoff::Real = 1e-8,
)
    full_free_stream_fourier_mpo_it = MPO(mpos.full_free_streaming_fourier_mpo; sites = sites_mpo)
    half_free_stream_fourier_mpo_it = MPO(mpos.half_free_streaming_fourier_mpo; sites = sites_mpo)

    x_fourier_mpo_it = MPO(mpos.x_fourier_mpo; sites = sites_mpo)
    x_inv_fourier_mpo_it = MPO(mpos.x_inv_fourier_mpo; sites = sites_mpo)

    v_fourier_mpo_it = MPO(mpos.v_fourier_mpo; sites = sites_mpo)
    v_inv_fourier_mpo_it = MPO(mpos.v_inv_fourier_mpo; sites = sites_mpo)

    stretched_kv_mpo_it = MPO(mpos.stretched_kv_mpo; sites = sites_mpo)

    diffusion_mpo_it = MPO(mpos.diffusion_mpo; sites = sites_mpo)
    half_filtering_mpo_it = MPO(mpos.half_filtering_mpo; sites = sites_mpo)
    #mixed_basis_free_stream_mpo_it = MPO(mpos.mixed_basis_free_streaming_mpo; sites = sites_mpo)

    full_free_stream_mpo_it = apply(
        x_inv_fourier_mpo_it,
        apply(
            full_free_stream_fourier_mpo_it,
            x_fourier_mpo_it;
            alg = "naive",
            cutoff = cutoff,
        );
        alg = "naive",
        cutoff = cutoff,
    )

    half_free_stream_mpo_it = apply(
        x_inv_fourier_mpo_it,
        apply(
            half_free_stream_fourier_mpo_it,
            x_fourier_mpo_it;
            alg = "naive",
            cutoff = cutoff,
        );
        alg = "naive",
        cutoff = cutoff,
    )

    if use_gpu
        full_free_stream_fourier_mpo_it = cu(full_free_stream_fourier_mpo_it)
        half_free_stream_fourier_mpo_it = cu(half_free_stream_fourier_mpo_it)

        full_free_stream_mpo_it = cu(full_free_stream_mpo_it)
        half_free_stream_mpo_it = cu(half_free_stream_mpo_it)

        x_fourier_mpo_it = cu(x_fourier_mpo_it)
        x_inv_fourier_mpo_it = cu(x_inv_fourier_mpo_it)

        v_fourier_mpo_it = cu(v_fourier_mpo_it)
        v_inv_fourier_mpo_it = cu(v_inv_fourier_mpo_it)

        stretched_kv_mpo_it = cu(stretched_kv_mpo_it)

        diffusion_mpo_it = cu(diffusion_mpo_it)
        half_filtering_mpo_it = cu(half_filtering_mpo_it)
        #mixed_basis_free_stream_mpo_it = cu(mixed_basis_free_stream_mpo_it)
    end

    return (
        half_free_stream_fourier = half_free_stream_fourier_mpo_it,
        full_free_stream_fourier = full_free_stream_fourier_mpo_it,
        full_free_stream = full_free_stream_mpo_it,
        half_free_stream = half_free_stream_mpo_it,
        x_fourier = x_fourier_mpo_it,
        x_inv_fourier = x_inv_fourier_mpo_it,
        v_fourier = v_fourier_mpo_it,
        v_inv_fourier = v_inv_fourier_mpo_it,
        stretched_kv = stretched_kv_mpo_it,
        diffusion = diffusion_mpo_it,
        half_filtering = half_filtering_mpo_it,
        #mixed_basis_free_stream = mixed_basis_free_stream_mpo_it,
    )
end
