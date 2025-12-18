using Plots
using LaTeXStrings

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
    directory_path = "results/landau_damping_TCI_v3/"
    filepath = joinpath(directory_path, "data.csv")
    steps, times, charges, ef_energy, kinetic_energy, total_energy = read_data(filepath)

    gamma_analytic = -0.15336
    #gamma_analytic = -0.2
    ef_analytic = [8e-2 * exp(2 * gamma_analytic * t) for t in times]

    default(
        fontfamily = "Computer Modern",
        guidefontsize = 14,
        tickfontsize = 12,
        legendfontsize = 12,
        linewidth = 2,
    )

    plt = plot(
        times,
        [ef_energy, ef_analytic],
        xlabel = L"t",
        ylabel = L"\mathcal{E}_E",
        label = [L"\mathrm{Numerical}" L"\mathrm{Analytic}"],
        legend = :bottomleft,
        linestyles = [:solid :dash],
        yaxis = :log10,
        framestyle = :box,
    )
    save_path = joinpath(directory_path, "energies_vs_time.png")
    savefig(plt, save_path)
end
