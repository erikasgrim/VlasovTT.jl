using Plots
using Plots.PlotMeasures
using Statistics

include(joinpath("plot_defaults.jl"))
PlotDefaults.apply!()

base_dir = "final_results/landau_damping/sweep_cutoff"

function read_runtime(filepath::String)
    steps = Int[]
    tci_time = Float64[]
    mpo_time = Float64[]
    step_time = Float64[]
    open(filepath, "r") do io
        readline(io)
        for line in eachline(io)
            cols = split(line, ",")
            push!(steps, parse(Int, cols[1]))
            push!(tci_time, parse(Float64, cols[2]))
            push!(mpo_time, parse(Float64, cols[3]))
            push!(step_time, parse(Float64, cols[4]))
        end
    end
    return (
        steps = steps,
        tci_time = tci_time,
        mpo_time = mpo_time,
        step_time = step_time,
    )
end

function cutoff_value(path::String)
    params_path = joinpath(path, "simulation_params.txt")
    if isfile(params_path)
        for line in eachline(params_path)
            occursin("SVD_tolerance", line) || continue
            _, val = strip.(split(line, "=", limit = 2))
            return strip(val)
        end
    end
    return "unknown"
end

case_paths = sort(filter(isdir, readdir(base_dir; join = true)))
plots = Plots.Plot[]

for (idx, path) in enumerate(case_paths)
    runtime_path = joinpath(path, "runtime.csv")
    data = read_runtime(runtime_path)
    case_name = basename(path)
    cutoff = cutoff_value(path)
    title_label = "$(case_name) (cutoff = $(cutoff))"

    p = plot(
        data.steps,
        data.tci_time,
        label = "TCI",
        xlabel = "Step",
        ylabel = idx == 1 ? "Runtime (s)" : "",
        title = title_label,
        titlelocation = :left,
        legend = idx == 1 ? :topright : false,
        left_margin = idx == 1 ? 2mm : 0mm,
    )
    plot!(p, data.steps, data.mpo_time; label = "TT/MPO contr.")
    plot!(p, data.steps, data.step_time; label = "Total")
    push!(plots, p)

    tci_mean = mean(data.tci_time)
    tci_std = std(data.tci_time)
    mpo_mean = mean(data.mpo_time)
    mpo_std = std(data.mpo_time)
    step_mean = mean(data.step_time)
    step_std = std(data.step_time)
    println("$(case_name) (cutoff = $(cutoff))")
    println("  TCI: $(tci_mean) +/- $(tci_std) s")
    println("  MPO: $(mpo_mean) +/- $(mpo_std) s")
    println("  Total: $(step_mean) +/- $(step_std) s")
end

plt = plot(plots...; layout = (1, 4), link = :y)

#savefig(plt, "plots/paper_figures/runtime_landau.pdf")
