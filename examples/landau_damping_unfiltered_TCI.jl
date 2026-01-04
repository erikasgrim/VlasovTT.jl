using Revise
using VlasovTT

using QuanticsGrids
using QuanticsTCI
import TensorCrossInterpolation as TCI

using ITensorMPS
using ITensors

using HDF5

using CUDA

using Plots
using ProgressBars
using Dates

Base.@kwdef struct LandauDampingConfig
    # Simulation parameters
    dt::Float64 = 0.1
    Tfinal::Float64 = 30.0
    simulation_name::String = "landau_damping"

    # Grid parameters
    R::Int = 12
    k_cut::Int = 2^8
    beta::Float64 = 10.0
    xmin::Float64 = -2pi
    xmax::Float64 = 2pi
    vmin::Float64 = -6.0
    vmax::Float64 = 6.0

    # TT parameters
    TCI_tolerance::Float64 = 1e-8
    maxrank::Int = 200
    maxrank_ef::Int = 16
    cutoff::Float64 = 1e-8
end

function config_to_namedtuple(config::LandauDampingConfig)
    return (; (name => getproperty(config, name) for name in propertynames(config))...)
end

function with_overrides(config::LandauDampingConfig, overrides::NamedTuple)
    return LandauDampingConfig(; config_to_namedtuple(config)..., overrides...)
end

function run_simulation(config::LandauDampingConfig; use_gpu::Bool = true, save_every::Int = 10)

    # Simulation parameters
    dt = config.dt
    Tfinal = config.Tfinal
    nsteps = Int(Tfinal / dt)

    simulation_name = config.simulation_name
    
    # Grid parameters
    R = config.R
    k_cut = config.k_cut # This keeps the 2^... lowest negative AND positive modes. 
    beta = config.beta

    xmin = config.xmin
    xmax = config.xmax
    vmin = config.vmin
    vmax = config.vmax

    # TT parameters
    TCI_tolerance = config.TCI_tolerance
    maxrank = config.maxrank
    maxrank_ef = config.maxrank_ef
    cutoff = config.cutoff

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
    )

    # Set up simulation directories
    simulation_dir = joinpath("results", simulation_name)
    figure_dir = joinpath(simulation_dir, "figures")
    mps_dir = joinpath(simulation_dir, "mps")
    data_filepath = joinpath(simulation_dir, "data.csv")
    bond_dims_filepath = joinpath(simulation_dir, "bond_dims.csv")
    if isfile(data_filepath)
        rm(data_filepath)
    end
    if isfile(bond_dims_filepath)
        rm(bond_dims_filepath)
    end
    mkpath(figure_dir)
    mkpath(mps_dir)
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

    open(bond_dims_filepath, "w") do io
        println(io, join(["bond_dim_$i" for i in 1:(length(psi_mps) - 1)], ","))
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
        open(bond_dims_filepath, "a") do io
            println(io, join(linkdims(psi_plot), ","))
        end
        write_data(
            step, 
            round(step * params.dt, digits=n_digits), 
            round(real(total_charge(psi_plot, phase, observables_cache)), digits=n_digits),
            round(real(electric_field_energy(ef_mps, phase)), digits=n_digits),
            round(real(ef_energy_first_mode), digits=n_digits),
            round(real(kinetic_energy(psi_plot, phase, observables_cache)), digits=n_digits),
            round(real(total_momentum(psi_plot, phase, observables_cache)), digits=n_digits),
            norm(psi_plot),
            maxlinkdim(psi_mps),
            elapsed_time,
            simulation_dir
        )

        if step == 1 || step % 50 == 0
            mps_path = joinpath(mps_dir, "psi_step$(step).h5")
            f = h5open(mps_path, "w")
            write(f, "psi", ITensors.cpu(psi_plot))
            close(f)
        end

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

function run_simulation_sweep(;
    base_config::LandauDampingConfig = LandauDampingConfig(),
    parameter::Symbol,
    values,
    sweep_name::Union{Nothing,String} = nothing,
    use_gpu::Bool = true,
    save_every::Int = 10,
)
    if !(parameter in fieldnames(LandauDampingConfig))
        error("Unknown parameter: $(parameter). Expected one of $(fieldnames(LandauDampingConfig)).")
    end

    sweep_stamp = sweep_name === nothing ? Dates.format(now(), "yyyymmdd_HHMMSS") : sweep_name
    sweep_dir = "sweep_" * sweep_stamp

    case_index = 1
    for value in values
        case_config = with_overrides(base_config, NamedTuple{(parameter,)}((value,)))
        case_base_name = case_config.simulation_name
        case_id = "case_" * lpad(case_index, 3, '0')
        case_config = with_overrides(
            case_config,
            (simulation_name = joinpath(case_base_name, sweep_dir, case_id),),
        )
        run_simulation(case_config; use_gpu = use_gpu, save_every = save_every)
        case_index += 1
    end
end

run_simulation_sweep(
    parameter = :cutoff,
    values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
    sweep_name = "cutoff",
)
