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
    directory_path = "results/landau_damping_TCI_v4/"
    filepath = joinpath(directory_path, "data.csv")
    steps, times, charges, ef_energy, kinetic_energy, total_energy = read_data(filepath)

    # Extract the indices of local maximima of the electric field energy
    ef_max_indices = findall(i -> (i > 1 && i < length(ef_energy)) && (ef_energy[i] > ef_energy[i-1]) && (ef_energy[i] > ef_energy[i+1]), 1:length(ef_energy))
    ef_max_times = times[ef_max_indices]
    log_ef_max_values = log.(ef_energy[ef_max_indices])
    matrix = hcat(2 .* ef_max_times, ones(length(ef_max_times)))
    fit = matrix \ log_ef_max_values
    ef_fitted = exp(fit[2]) .* exp.(2 * fit[1] .* times)

    # Linear fit 
    gamma_analytic = -0.15139
    ef_analytic = [ef_fitted[1] * exp(2 * gamma_analytic * t) for t in times]

    default(
        fontfamily = "Computer Modern",
        guidefontsize = 14,
        tickfontsize = 12,
        legendfontsize = 12,
        linewidth = 2,
    )

    plt = plot(
        times,
        [ef_energy, ef_analytic, ef_fitted],
        xlabel = L"t\ [\omega_{pe}^{-1}]",
        ylabel = L"\mathcal{E}_E",
        label = [L"\mathrm{Numerical}" L"\mathrm{Analytic\ }\gamma=-0.151" L"\mathrm{Fitted\ \gamma=}" * latexstring(round(fit[1]; digits=4))],
        legend = :bottomleft,
        linestyles = [:solid :dash :dot],
        color = [1 2 :black],
        yaxis = :log10,
        framestyle = :box,
    )

    # add scatter plot of local maxima
    scatter!(
        ef_max_times,
        ef_energy[ef_max_indices],
        label = nothing,
        markershape = :cross,
        color = :black,
    )

    save_path = joinpath(directory_path, "energies_vs_time.png")
    savefig(plt, save_path)
end
