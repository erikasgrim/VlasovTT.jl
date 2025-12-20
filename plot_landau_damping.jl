using Plots
using LaTeXStrings
using VlasovTT: read_data

let 
    directory_path = "results/landau_damping_TCI_v5/"
    filepath = joinpath(directory_path, "data.csv")
    data = read_data(filepath)
    times = data.times
    ef_energy = data.ef_energy
    ef_energy_mode1 = data.ef_energy_mode1

    # Extract the indices of local maximima of the electric field energy
    ef_max_indices = findall(i -> (i > 1 && i < length(ef_energy_mode1)) && (ef_energy_mode1[i] > ef_energy_mode1[i-1]) && (ef_energy_mode1[i] > ef_energy_mode1[i+1]), 1:length(ef_energy))
    ef_max_times = times[ef_max_indices]
    log_ef_max_values = log.(ef_energy_mode1[ef_max_indices])
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
        [ef_energy_mode1, ef_analytic],
        xlabel = L"t\ [\omega_{pe}^{-1}]",
        ylabel = L"\mathcal{E}_1",
        label = [L"\mathrm{Numerical}" L"\mathrm{Analytic\ }\gamma=-0.151"],
        legend = :bottomleft,
        linestyles = [:solid :dot],
        color = [1 2 :black],
        yaxis = :log10,
        framestyle = :box,
    )

    # add scatter plot of local maxima
    scatter!(
        ef_max_times,
        ef_energy_mode1[ef_max_indices],
        label = nothing,
        markershape = :cross,
        color = :black,
    )

    bonddim_plt = plot(
        times,
        data.bond_dimensions,
        xlabel = L"t\ [\omega_{pe}^{-1}]",
        ylabel = L"\chi",
        label = L"\mathrm{Numerical}",
        legend = :bottomright,
        framestyle = :box,
    )

    savefig(plt, joinpath(directory_path, "energies_vs_time.png"))
    savefig(bonddim_plt, joinpath(directory_path, "bond_dimension_vs_time.png"))
end
