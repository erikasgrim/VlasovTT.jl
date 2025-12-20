using Revise
using VlasovTT

using QuanticsGrids
using QuanticsTCI
import TensorCrossInterpolation as TCI

using ITensorMPS
using ITensors

using CUDA

using Plots
using ProgressBars

function run_two_stream(; use_gpu::Bool = true, save_every::Int = 10)

    # Simulation parameters
    dt = .1
    Tfinal = 60.0
    nsteps = Int(Tfinal / dt)
    k_cut = 2^8 # This keeps the 2^... lowest negative AND positive modes. 
    beta = 2.0

    simulation_name = "landau_damping_TCI_v5"
    
    # Grid parameters
    R = 10

    xmin = -2pi
    xmax = 2pi
    vmin = -6.0
    vmax = 6.0
    normalize = true

    # TT parameters
    TCI_tolerance = 1e-8
    maxrank = 128
    maxrank_ef = 12
    cutoff = 1e-8

    # Build phase space grids and simulation parameters
    phase = PhaseSpaceGrids(R, xmin, xmax, vmin, vmax)
    params = VlasovTT.SimulationParams(
        dt = dt,
        tolerance = TCI_tolerance,
        cutoff = cutoff,
        maxrank = maxrank,
        maxrank_ef = maxrank_ef,
        k_cut = k_cut,
        beta = beta,
        use_gpu = use_gpu,
        alg = "naive",
        l1_normalize = normalize,
    )

    # Set up simulation directories
    simulation_dir = joinpath("results", simulation_name)
    figure_dir = joinpath(simulation_dir, "figures")
    data_filepath = joinpath(simulation_dir, "data.csv")
    if isfile(data_filepath)
        rm(data_filepath)
    end
    mkpath(figure_dir)
    write_parameters(params, phase, simulation_dir)

    # Build solver MPOs
    solver_mpos = build_solver_mpos(
        phase;
        dt = params.dt,
        tolerance = params.tolerance,
        k_cut = params.k_cut,
        beta = params.beta,
        eps0 = 1.0,
        v0 = params.v0,
        lsb_first = true,
    )

    # Build initial pivots around +- v0 in velocity space
    ic_fn = linear_landau_damping_ic(phase; A = 0.1, v0 = 0.0, vt = 1.0, mode = 1)
    tt, _, _ = build_initial_tt(ic_fn, R; tolerance = params.tolerance)
    println("Initial condition TT ranks: ", TCI.rank(tt))

    # Convert to ITensor MPS & MPOs
    psi_mps = MPS(TCI.TensorTrain(tt))
    mps_sites = siteinds(psi_mps)
    sites_mpo = [[prime(s, 1), s] for s in mps_sites]

    itensor_mpos = prepare_itensor_mpos(
        solver_mpos,
        sites_mpo;
        use_gpu = params.use_gpu,
    )

    if params.use_gpu
        psi_mps = cu(psi_mps)
    end

    # Values for plotting
    x_vals = range(phase.xmin, phase.xmax; length = min(300, phase.M))
    v_vals = range(phase.vmin, phase.vmax; length = min(300, phase.M))

    # Build cache for observables
    observables_cache = build_observables_cache(psi_mps, phase)

    # Half step in free streaming
    psi_mps = apply(itensor_mpos.x_fourier, psi_mps; alg = params.alg, maxdim = maxrank, cutoff = params.cutoff)
    psi_mps = apply(itensor_mpos.half_free_stream_fourier, psi_mps; alg = params.alg, maxdim = maxrank, cutoff = params.cutoff)
    psi_mps = apply(itensor_mpos.x_inv_fourier, psi_mps; alg = params.alg, maxdim = maxrank, cutoff = params.cutoff)
    psi_mps = apply(itensor_mpos.v_fourier, psi_mps; alg = params.alg, maxdim = maxrank, cutoff = params.cutoff)

    previous_tci = nothing
    loop_start_time = time()
    iter = ProgressBar(1:nsteps)
    for step in iter
        psi_mps, ef_mps, previous_tci = strang_step_unfiltered_TCI!(
            psi_mps,
            phase,
            solver_mpos.full_poisson_mpo,
            itensor_mpos.full_free_stream_fourier,
            itensor_mpos.x_fourier,
            itensor_mpos.x_inv_fourier,
            itensor_mpos.v_fourier,
            itensor_mpos.v_inv_fourier,
            mps_sites,
            previous_tci;
            params = params,
        )

        ef_energy_first_mode = electric_field_mode_energy(
           ef_mps,
           phase,
           MPO(solver_mpos.fourier_mpo; sites = [[prime(s, 1), s] for s in siteinds(ef_mps)]);
        )

        n_digits = 12
        elapsed_time = round(time() - loop_start_time; digits = n_digits)
        psi_plot = copy(psi_mps)
        psi_plot = apply(itensor_mpos.v_inv_fourier, psi_plot; alg = params.alg, maxdim = maxrank, cutoff = params.cutoff)
        write_data(
            step, 
            round(step * params.dt, digits=n_digits), 
            round(real(total_charge(psi_plot, phase, observables_cache)), digits=n_digits),
            round(real(electric_field_energy(ef_mps, phase)), digits=n_digits),
            round(real(ef_energy_first_mode), digits=n_digits),
            round(real(kinetic_energy(psi_plot, phase, observables_cache)), digits=n_digits),
            round(real(total_momentum(psi_plot, phase, observables_cache)), digits=n_digits),
            maxlinkdim(psi_mps),
            elapsed_time,
            simulation_dir
        )

        if step % save_every == 0 || step == 1 || step == nsteps
            tt_snapshot = TCI.TensorTrain(ITensors.cpu(psi_plot))
            f_vals = [tt_snapshot(origcoord_to_quantics(phase.x_v_grid, (x, v))) for v in v_vals, x in x_vals]
            Plots.savefig(
                Plots.heatmap(
                    x_vals,
                    v_vals,
                    abs.(f_vals);
                    xlabel = "x",
                    ylabel = "v",
                    title = "t = $(round(step * params.dt, digits = 3))",
                ),
                "results/$simulation_name/figures/phase_space_step$(step).png",
            )

            ef_snapshot = TCI.TensorTrain(ITensors.cpu(ef_mps))
            E_x_vals = [real.(ef_snapshot(origcoord_to_quantics(phase.x_grid, x))) for x in x_vals]
            Plots.savefig(
                Plots.plot(
                    x_vals,
                    E_x_vals;
                    xlabel = "x",
                    ylabel = "E(x)",
                    title = "Electric Field at t = $(round(step * params.dt, digits = 3))",
                    ylim = (-0.1, 0.1),
                ),
                "results/$simulation_name/figures/electric_field_step$(step).png",
            )
        end
    end

    return psi_mps
end

run_two_stream()
