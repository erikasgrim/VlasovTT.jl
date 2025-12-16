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
        println(io, "Grid Parameters:")
        println(io, "R = $(phase.R)")
        println(io, "xmin = $(phase.xmin)")
        println(io, "xmax = $(phase.xmax)")
        println(io, "vmin = $(phase.vmin)")
        println(io, "vmax = $(phase.vmax)")
    end
end

function write_data(step::Int, time::Real, charge::Real, ef_energy::Real, k_energy::Real, bond_dimension::Int, elapsed_time::Real, directory::String)
    filepath = joinpath(directory, "data.csv")
    needs_header = !isfile(filepath) || filesize(filepath) == 0
    open(filepath, "a") do io
        if needs_header
            println(io, "step,time,charge,ef_energy,kinetic_energy,total_energy,bond_dimension,elapsed_time")
        end
        total_energy = ef_energy + k_energy
        println(io, "$(step),$(time),$(charge),$(ef_energy),$(k_energy),$(total_energy),$(bond_dimension),$(elapsed_time)")
    end
end
