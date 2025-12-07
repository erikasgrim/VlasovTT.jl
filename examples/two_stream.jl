using VlasovTT
using QuanticsGrids
using QuanticsTCI
import TensorCrossInterpolation as TCI
using ITensorMPS
using ITensors
using CUDA
using Plots
using ProgressBars

function write_parameters(params::SimulationParams, phase, directory::String)
    mkpath(directory)
    open(joinpath(directory, "simulation_params.txt"), "w") do io
        println(io, "Simulation Parameters:")
        println(io, "dt = $(params.dt)")
        println(io, "tolerance = $(params.tolerance)")
        println(io, "maxrank = $(params.maxrank)")
        println(io, "k_cut = $(params.k_cut)")
        println(io, "beta = $(params.beta)")
        println(io, "Grid Parameters:")
        println(io, "R = $(phase.R)")
        println(io, "xmin = $(phase.xmin)")
        println(io, "xmax = $(phase.xmax)")
        println(io, "vmin = $(phase.vmin)")
        println(io, "vmax = $(phase.vmax)")
    end
end

function write_data(step::Int, time::Real, charge::Real, ef_energy::Real, k_energy::Real, directory::String)
    filepath = joinpath(directory, "data.csv")
    needs_header = !isfile(filepath) || filesize(filepath) == 0
    open(filepath, "a") do io
        if needs_header
            println(io, "step,time,charge,ef_energy,kinetic_energy,total_energy")
        end
        println(io, "$(step),$(time),$(charge),$(ef_energy),$(k_energy),$(ef_energy + k_energy)")
    end
end

function run_two_stream(; use_gpu::Bool = true, save_every::Int = 10)
    
    dt = 1e-3
    Tfinal = 20.0
    nsteps = Int(Tfinal / dt)

    R = 8

    xmin = -10.0
    xmax = 10.0
    vmin = -6.0
    vmax = 6.0

    tolerance = 1e-9
    maxrank = 64
    k_cut = 2^8
    beta = 0.0

    phase = PhaseSpaceGrids(R, xmin, xmax, vmin, vmax)

    params = SimulationParams(
        dt = dt,
        tolerance = tolerance,
        maxrank = maxrank,
        k_cut = k_cut,
        beta = beta,
        use_gpu = use_gpu,
        alg = "naive",
    )

    simulation_name = "two_stream"
    simulation_dir = joinpath("results", simulation_name)
    figure_dir = joinpath(simulation_dir, "figures")
    data_filepath = joinpath(simulation_dir, "data.csv")
    if isfile(data_filepath)
        rm(data_filepath)
    end
    mkpath(figure_dir)
    write_parameters(params, phase, simulation_dir)

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
    init_charge = total_charge(psi_mps, phase)
    init_ke = kinetic_energy(psi_mps, phase)
    println("Initial norm: $init_norm")
    println("Initial total charge: $init_charge")
    println("Initial kinetic energy: $init_ke")
    write_data(0, 0.0, init_charge, 0.0, init_ke, simulation_dir)

    accel_cache = AccelerationTCICache()
    observables_cache = build_observables_cache(psi_mps, phase)
    iter = ProgressBar(1:nsteps)
    for step in iter
        psi_mps, ef_mps = strang_step!(
            psi_mps,
            phase,
            solver_mpos.full_poisson_mpo,
            itensor_mpos.free_stream,
            itensor_mpos.v_fourier,
            itensor_mpos.v_inv_fourier,
            sites_mpo;
            params = params,
            accel_cache = accel_cache,
            return_field = true,
            target_norm = init_charge,
            reuse_strategy = :resweep,
        )

        if step % save_every == 0
            set_description(iter, "Norm ratio: $(round(norm(psi_mps) / init_norm, digits = 6)), Bond dim: $(maxlinkdim(psi_mps))")
            n_digits = 4
            write_data(
                step, 
                round(step * params.dt, digits=n_digits), 
                round(real(total_charge(psi_mps, phase; cache=observables_cache)), digits=n_digits),
                round(real(electric_field_energy(ef_mps, phase)), digits=n_digits),
                round(real(kinetic_energy(psi_mps, phase; cache=observables_cache)), digits=n_digits),
                simulation_dir
            )
            
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
                "results/$simulation_name/figures/phase_space_step$(step).png",
            )
        end
    end

    return psi_mps
end

run_two_stream()
