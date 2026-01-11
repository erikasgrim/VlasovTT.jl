using Plots
using Plots.PlotMeasures
using LaTeXStrings
using VlasovTT: read_data

include(joinpath("plot_defaults.jl"))
PlotDefaults.apply!()

ts_dir = "final_results/two_stream/sweep_cutoff/"
landau_dir = "final_results/landau_damping/sweep_cutoff/"

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
                label = LaTeXString("\\epsilon = $(strip(val))")
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

p1 = plot(
    ylabel = L"|\Delta E| / |E_0|",
    title = "(a)",
    titlelocation = :left,
    legend = nothing,
    xticks = :none,
    yaxis = :log10,
    left_margin = 2mm,
)
for s in landau_sets
    times, values = drop_first(s.data.times, rel_dev(s.data.total_energy))
    plot!(p1, times, values; label = s.name)
end

p2 = plot(
    title = "(b)",
    titlelocation = :left,
    legend = :topright,
    xticks = :none,
    yaxis = :log10,
)
for s in ts_sets
    times, values = drop_first(s.data.times, rel_dev(s.data.total_energy))
    plot!(p2, times, values; label = s.name)
end

p3 = plot(
    ylabel = L"|\Delta P|",
    title = "(c)",
    titlelocation = :left,
    legend = nothing,
    xticks = :none,
    yaxis = :log10,
    left_margin = 2mm,
)
for s in landau_sets
    times, values = drop_first(s.data.times, abs_dev(s.data.momentum))
    plot!(p3, times, values; label = s.name)
end

p4 = plot(
    title = "(d)",
    titlelocation = :left,
    legend = nothing,
    xticks = :none,
    yaxis = :log10,
)
for s in ts_sets
    times, values = drop_first(s.data.times, abs_dev(s.data.momentum))
    plot!(p4, times, values; label = s.name)
end

plt = plot(p1, p2, p3, p4; layout = (2, 2), link = :x)

savefig(plt, "plots/paper_figures/conservation.pdf")
