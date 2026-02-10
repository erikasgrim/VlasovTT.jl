using Plots
using Plots.PlotMeasures
using LaTeXStrings
using VlasovTT: read_data

include(joinpath("plot_defaults.jl"))
PlotDefaults.apply!()

ts_dir = "final_results/two_stream/sweep_cutoff/"
landau_dir = "final_results/landau_damping/sweep_cutoff/"

cutoff_labels = Dict(1e-7 => "10^{-7}", 1e-10 => "10^{-10}")

function cutoff_label(cutoff)
    for (key, label) in cutoff_labels
        if isfinite(cutoff) && isapprox(cutoff, key; rtol = 1e-12, atol = 0.0)
            return label
        end
    end
    if isfinite(cutoff) && cutoff > 0
        exponent = round(Int, log10(cutoff))
        return "10^{$(exponent)}"
    end
    return "NaN"
end

function read_sweep(dir)
    entries = sort(filter(isdir, readdir(dir; join = true)))
    datasets = []
    for path in entries
        data_path = joinpath(path, "data.csv")
        isfile(data_path) || continue
        params_path = joinpath(path, "simulation_params.txt")
        label = basename(path)
        if isfile(params_path)
            for line in eachline(params_path)
                occursin("SVD_tolerance", line) || continue
                _, val = strip.(split(line, "=", limit = 2))
                cutoff = parse(Float64, strip(val))
                label = latexstring("\$\\epsilon = $(cutoff_label(cutoff))\$")
                break
            end
        end
        push!(datasets, (name = label, data = read_data(data_path)))
    end
    return datasets
end

landau_sets = read_sweep(landau_dir)
ts_sets = read_sweep(ts_dir)

p1 = plot(
    xlabel = "",
    ylabel = L"\mathcal{E}_{\mathrm{field}}^{(1)}(t)",
    yaxis = :log10,
    title = "(a)",
    titlelocation = :left,
    legend = nothing,
    xticks = :none,
)
for s in landau_sets
    plot!(p1, s.data.times, s.data.ef_energy_mode1; label = s.name)
end

p2 = plot(
    xlabel = "",
    yaxis = :log10,
    title = "(b)",
    titlelocation = :left,
    legend = nothing,
    xticks = :none,
)
for s in ts_sets
    plot!(p2, s.data.times, s.data.ef_energy_mode1; label = s.name)
end

p3 = plot(
    xlabel = L"t\ [\omega_{pe}^{-1}]",
    ylabel = L"$\chi_{\mathrm{max}}$",
    title = "(c)",
    titlelocation = :left,
    legend = nothing,
)
for s in landau_sets
    plot!(p3, s.data.times, s.data.bond_dimensions; label = s.name)
end

p4 = plot(
    xlabel = L"t\ [\omega_{pe}^{-1}]",
    title = "(d)",
    titlelocation = :left,
    legend = :bottomright,
)
for s in ts_sets
    plot!(p4, s.data.times, s.data.bond_dimensions; label = s.name)
end

plt = plot(p1, p2, p3, p4; layout = (2, 2), left_margin = 2mm, bottom_margin = 2mm, size = (833, 600))

savefig(plt, "plots/paper_figures/truncation_effects.pdf")
