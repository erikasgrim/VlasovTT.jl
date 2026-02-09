using Plots
using LaTeXStrings
using VlasovTT: read_data

let 
    directory_path = "results/two_stream_unfiltered_TCI_v15/"
    filepath = joinpath(directory_path, "data.csv")
    data = read_data(filepath)
    times = data.times
    ef_energy = data.ef_energy

    # Linear fit     
    ef_indices = findall(i -> (times[i] > 10.0) && (times[i] < 15), 1:length(ef_energy))
    ef_times = times[ef_indices]
    log_ef_values = log.(ef_energy[ef_indices])
    matrix = hcat(2 .* ef_times, ones(length(ef_times)))
    fit = matrix \ log_ef_values
    ef_fitted = exp(fit[2]) .* exp.(2 * fit[1] .* times)

    default(
        fontfamily = "Computer Modern",
        guidefontsize = 14,
        tickfontsize = 12,
        legendfontsize = 12,
        linewidth = 2,
    )

    linear_indices = findall(i -> (times[i] > 1.5) && (times[i] < 17.0), 1:length(ef_energy))
    gamma_analytic = 0.35355
    ef_analytic = [exp(fit[2]) .* exp(2 * gamma_analytic * t) for t in times]

    plt = plot(
        times,
        ef_energy,
        xlabel = L"t\ [\omega_{pe}^{-1}]",
        ylabel = L"\mathcal{E}_1",
        label = L"\mathrm{Numerical}",
        legend = :bottomright,
        yaxis = :log10,
        framestyle = :box,
    )

    plot!(
        plt,
        times[linear_indices],
        ef_analytic[linear_indices],
        linestyles = :dot,
        label = L"\mathrm{Analytic\ (\gamma = 0.3535)}",
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
