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

function rel_dev(values)
    ref = values[1]
    return abs.(values .- ref) ./ abs(ref)
end

function abs_dev(values)
    ref = values[1]
    return abs.(values .- ref)
end

function drop_first(times, values)
    if length(times) <= 1
        return times, values
    end
    return times[2:end], values[2:end]
end

landau_sets = read_sweep(landau_dir)
ts_sets = read_sweep(ts_dir)
positive_min(values) = minimum(filter(>(0), values))

energy_landau = vcat([rel_dev(s.data.total_energy)[2:end] for s in landau_sets]...)
energy_ts = vcat([rel_dev(s.data.total_energy)[2:end] for s in ts_sets]...)
row1_limits = (min(positive_min(energy_landau), positive_min(energy_ts)),
    max(maximum(energy_landau), maximum(energy_ts)))

momentum_landau = vcat([abs_dev(s.data.momentum)[2:end] for s in landau_sets]...)
momentum_ts = vcat([abs_dev(s.data.momentum)[2:end] for s in ts_sets]...)
row2_limits = (min(positive_min(momentum_landau), positive_min(momentum_ts)),
    max(maximum(momentum_landau), maximum(momentum_ts)))

p1 = plot(
    ylabel = L"|E - E_0|/|E_0|",
    title = "(a)",
    titlelocation = :left,
    legend = nothing,
    xticks = :none,
    yaxis = :log10,
    ylims = row1_limits,
    left_margin = 2mm,
)
for s in landau_sets
    times, values = drop_first(s.data.times, rel_dev(s.data.total_energy))
    plot!(p1, times, values; label = s.name)
end

p2 = plot(
    title = "(b)",
    titlelocation = :left,
    legend = nothing,
    xticks = :none,
    yaxis = :log10,
    ylims = row1_limits,
)
for s in ts_sets
    times, values = drop_first(s.data.times, rel_dev(s.data.total_energy))
    plot!(p2, times, values; label = s.name)
end

p3 = plot(
    xlabel = L"t\ [\omega_{pe}^{-1}]",
    ylabel = L"|\Delta P|",
    title = "(c)",
    titlelocation = :left,
    legend = nothing,
    xticks = :none,
    yaxis = :log10,
    ylims = row2_limits,
    left_margin = 2mm,
)
for s in landau_sets
    times, values = drop_first(s.data.times, abs_dev(s.data.momentum))
    plot!(p3, times, values; label = s.name)
end

p4 = plot(
    xlabel = L"t\ [\omega_{pe}^{-1}]",
    title = "(d)",
    titlelocation = :left,
    legend = :bottomright,
    xticks = :none,
    yaxis = :log10,
    ylims = row2_limits,
)
for s in ts_sets
    times, values = drop_first(s.data.times, abs_dev(s.data.momentum))
    plot!(p4, times, values; label = s.name)
end

plt = plot(p1, p2, p3, p4; layout = (2, 2), link = :x, bottom_margin = 2mm, size = (833, 600))

savefig(plt, "plots/paper_figures/conservation.pdf")
