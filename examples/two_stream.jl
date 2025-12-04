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
    mkpath("figures/two_stream")

    dt = 1e-2
    Tfinal = 10.0
    nsteps = Int(Tfinal / dt)

    R = 12

    xmin = -10.0
    xmax = 10.0
    vmin = -6.0
    vmax = 6.0

    phase = PhaseSpaceGrids(R, xmin, xmax, vmin, vmax)

    params = SimulationParams(
        dt = dt,
        tolerance = 1e-6,
        maxrank = 16,
        k_cut = 2^6,
        beta = 2.0,
        use_gpu = use_gpu,
        alg = "naive",
    )

    solver_mpos = build_solver_mpos(
        phase;
        dt = params.dt,
        tolerance = params.tolerance,
        k_cut = params.k_cut,
        beta = params.beta,
        eps0 = 1.0,
        lsb_first = true,
    )
    println("Full free streaming MPO ranks: ", TCI.rank(solver_mpos.full_free_streaming_mpo))
    println("Full Poisson MPO ranks: ", TCI.rank(solver_mpos.full_poisson_mpo))

    ic_fn = two_stream_instability_ic(phase; A = 0.1, v0 = 1.0, vt = 0.3, mode = 3)
    tt, interp_rank, interp_error = build_initial_tt(ic_fn, R; tolerance = params.tolerance)
    println("Initial condition TT ranks: ", TCI.rank(tt))

    psi_mps = MPS(tt)
    sites_mps = siteinds(psi_mps)
    sites_mpo = [[prime(s, 1), s] for s in sites_mps]

    itensor_mpos = prepare_itensor_mpos(
        solver_mpos,
        sites_mpo;
        use_gpu = params.use_gpu,
    )

    if params.use_gpu
        psi_mps = cu(psi_mps)
    end

    x_vals = range(phase.xmin, phase.xmax; length = 300)
    v_vals = range(phase.vmin, phase.vmax; length = 300)

    init_norm = norm(psi_mps)

    iter = ProgressBar(1:nsteps)
    for step in iter
        psi_mps = strang_step!(
            psi_mps,
            phase,
            solver_mpos.full_poisson_mpo,
            itensor_mpos.free_stream,
            itensor_mpos.v_fourier,
            itensor_mpos.v_inv_fourier,
            sites_mpo;
            params = params,
            target_norm=init_norm,
        )

        if step % save_every == 0
            set_description(iter, "Norm ratio: $(round(norm(psi_mps) / init_norm, digits = 6))")

            tt_snapshot = TCI.TensorTrain(ITensors.cpu(psi_mps))
            #println("Plotting step $step, TT ranks: ", TCI.rank(tt_snapshot))
            f_vals = [tt_snapshot(origcoord_to_quantics(phase.x_v_grid, (x, v))) for v in v_vals, x in x_vals]
            #println("Max f value at step $step: ", maximum(abs.(f_vals)))
            Plots.savefig(
                Plots.heatmap(
                    x_vals,
                    v_vals,
                    abs.(f_vals);
                    xlabel = "x",
                    ylabel = "v",
                    title = "t = $(round(step * params.dt, digits = 3))",
                ),
                "figures/two_stream/phase_space_step$(step).png",
            )
        end
    end

    return psi_mps
end

run_two_stream()
