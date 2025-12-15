using Plots

function read_data(filepath::String)
    steps = Int[]
    times = Float64[]
    charges = Float64[]
    ef_energy = Float64[]
    kinetic_energy = Float64[]
    total_energy = Float64[]
    open(filepath, "r") do io
        header = readline(io)  # Skip header
        for line in eachline(io)
            cols = split(line, ",")
            push!(steps, parse(Int, cols[1]))
            push!(times, parse(Float64, cols[2]))
            push!(charges, parse(Float64, cols[3]))
            push!(ef_energy, parse(Float64, cols[4]))
            push!(kinetic_energy, parse(Float64, cols[5]))
            push!(total_energy, parse(Float64, cols[6]))
        end
    end
    return steps, times, charges, ef_energy, kinetic_energy, total_energy
end

let 
    filepath = "results/two_stream/data.csv"
    steps, times, charges, ef_energy, kinetic_energy, total_energy = read_data(filepath)

    println(steps)

    plt = plot(
        times,
        [ef_energy, kinetic_energy, total_energy],
        xlabel = "Time",
        ylabel = "Total Charge",
        title = "Total Charge vs Time",
        legend = false,
    )
    savefig(plt, "results/two_stream/total_charge_vs_time.png")
end