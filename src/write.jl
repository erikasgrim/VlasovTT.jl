function write_parameters(params::VlasovTT.SimulationParams, phase, directory::String)
    mkpath(directory)
    open(joinpath(directory, "simulation_params.txt"), "w") do io
        println(io, "Simulation Parameters:")
        println(io, "dt = $(params.dt)")
        println(io, "tolerance = $(params.tolerance)")
        println(io, "SVD_tolerance = $(params.cutoff)")
        println(io, "maxrank = $(params.maxrank)")
        println(io, "maxrank_ef = $(params.maxrank_ef)")
        println(io, "k_cut = $(params.k_cut)")
        println(io, "beta = $(params.beta)")
        println(io, "v0 = $(params.v0)")
        println(io, "Grid Parameters:")
        println(io, "R = $(phase.R)")
        println(io, "xmin = $(phase.xmin)")
        println(io, "xmax = $(phase.xmax)")
        println(io, "vmin = $(phase.vmin)")
        println(io, "vmax = $(phase.vmax)")
    end
end

function write_data(
    step::Int, 
    time::Real, 
    charge::Real, 
    ef_energy::Real, 
    ef_energy_mode1::Real, 
    ke_energy::Real,
    momentum::Real, 
    tt_norm::Real,
    bond_dimension::Int, 
    elapsed_time::Real, 
    directory::String
)
    filepath = joinpath(directory, "data.csv")
    needs_header = !isfile(filepath) || filesize(filepath) == 0
    open(filepath, "a") do io
        if needs_header
            println(io, "step,time,charge,ef_energy,ef_energy_mode1,kinetic_energy,total_energy,momentum,tt_norm,bond_dimension,elapsed_time")
        end
        total_energy = ef_energy + ke_energy
        println(io, "$(step),$(time),$(charge),$(ef_energy),$(ef_energy_mode1),$(ke_energy),$(total_energy),$(momentum),$(tt_norm),$(bond_dimension),$(elapsed_time)")
    end
end

function write_runtimes(
    step::Int,
    tci_time::Real,
    mpo_time::Real,
    step_time::Real,
    directory::String,
)
    filepath = joinpath(directory, "runtime.csv")
    needs_header = !isfile(filepath) || filesize(filepath) == 0
    open(filepath, "a") do io
        if needs_header
            println(io, "step,tci_time,mpo_time,step_time")
        end
        println(io, "$(step),$(tci_time),$(mpo_time),$(step_time)")
    end
end

function read_data(filepath::String)
    steps = Int[]
    times = Float64[]
    charges = Float64[]
    ef_energy = Float64[]
    ef_energy_mode1 = Float64[]
    kinetic_energy = Float64[]
    total_energy = Float64[]
    momentum = Float64[]
    tt_norms = Float64[]
    bond_dimensions = Int[]
    elapsed_times = Float64[]
    open(filepath, "r") do io
        header = readline(io)  # Skip header
        for line in eachline(io)
            cols = split(line, ",")
            push!(steps, parse(Int, cols[1]))
            push!(times, parse(Float64, cols[2]))
            push!(charges, parse(Float64, cols[3]))
            push!(ef_energy, parse(Float64, cols[4]))
            push!(ef_energy_mode1, parse(Float64, cols[5]))
            push!(kinetic_energy, parse(Float64, cols[6]))
            push!(total_energy, parse(Float64, cols[7]))
            push!(momentum, parse(Float64, cols[8]))
            push!(tt_norms, parse(Float64, cols[9]))
            push!(bond_dimensions, parse(Int, cols[10]))
            push!(elapsed_times, parse(Float64, cols[11]))
        end
    end
    return (
        steps = steps,
        times = times,
        charges = charges,
        ef_energy = ef_energy,
        ef_energy_mode1 = ef_energy_mode1,
        kinetic_energy = kinetic_energy,
        total_energy = total_energy,
        momentum = momentum,
        tt_norms = tt_norms,
        bond_dimensions = bond_dimensions,
        elapsed_times = elapsed_times,
    )
end
