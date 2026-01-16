using Plots
using Plots.PlotMeasures
using HDF5
using ITensorMPS
using ITensors
using LaTeXStrings
using QuanticsGrids
using QuanticsTCI
using VlasovTT: PhaseSpaceGrids, read_data

include(joinpath("plot_defaults.jl"))
PlotDefaults.apply!()

landau_dir = "final_results/landau_damping/sweep_cutoff"
ts_dir = "final_results/two_stream/sweep_cutoff"

function read_grid_params(path)
    params = Dict{String,Float64}()
    for line in eachline(path)
        occursin("=", line) || continue
        key, val = strip.(split(line, "=", limit = 2))
        if key in ("R", "xmin", "xmax", "vmin", "vmax")
            params[key] = parse(Float64, val)
        end
    end
    R = Int(params["R"])
    return R, params["xmin"], params["xmax"], params["vmin"], params["vmax"]
end

function read_cutoff(path)
    for line in eachline(path)
        occursin("SVD_tolerance", line) || continue
        _, val = strip.(split(line, "=", limit = 2))
        return parse(Float64, strip(val))
    end
    return NaN
end

function fixed_mps_path(mps_dir::String, step::Int)
    path = joinpath(mps_dir, "psi_step$(step).h5")
    return path, step
end

function load_distribution(case_dir::String; nx::Int = 300, nv::Int = 300)
    params_path = joinpath(case_dir, "simulation_params.txt")
    data_path = joinpath(case_dir, "data.csv")
    mps_dir = joinpath(case_dir, "mps")
    mps_path, step = fixed_mps_path(mps_dir, 300)

    R, xmin, xmax, vmin, vmax = read_grid_params(params_path)
    phase = PhaseSpaceGrids(R, xmin, xmax, vmin, vmax)

    x_vals = range(phase.xmin, phase.xmax; length = nx)
    v_vals = range(phase.vmin, phase.vmax; length = nv)

    println("Loading MPS from: $mps_path")
    psi_mps = h5open(mps_path, "r") do file
        read(file, "psi", MPS)
    end
    tt_snapshot = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps))
    f_vals = [tt_snapshot(origcoord_to_quantics(phase.x_v_grid, (x, v))) for v in v_vals, x in x_vals]

    t_final = NaN
    if isfile(data_path)
        data = read_data(data_path)
        if !isempty(data.times)
            t_final = data.times[end]
        end
    end

    cutoff = isfile(params_path) ? read_cutoff(params_path) : NaN

    return (
        x_vals = x_vals,
        v_vals = v_vals,
        f_vals = f_vals,
        step = step,
        t_final = t_final,
        cutoff = cutoff,
    )
end

function load_cases(base_dir::String)
    entries = sort(filter(isdir, readdir(base_dir; join = true)))
    datasets = []
    for path in entries
        params_path = joinpath(path, "simulation_params.txt")
        isfile(params_path) || continue
        cutoff = read_cutoff(params_path)
        push!(datasets, (cutoff = cutoff, path = path))
    end
    sort!(datasets; by = d -> d.cutoff)
    return datasets
end

landau_sets = load_cases(landau_dir)
ts_sets = load_cases(ts_dir)
if length(landau_sets) >= 2
    landau_sets = [last(landau_sets), first(landau_sets)]
end
if length(ts_sets) >= 2
    ts_sets = [last(ts_sets), first(ts_sets)]
end

landau_sets = [merge(s, load_distribution(s.path)) for s in landau_sets]
ts_sets = [merge(s, load_distribution(s.path)) for s in ts_sets]

landau_clim = (
    minimum(minimum(abs.(s.f_vals)) for s in landau_sets),
    maximum(maximum(abs.(s.f_vals)) for s in landau_sets),
)
ts_clim = (
    minimum(minimum(abs.(s.f_vals)) for s in ts_sets),
    maximum(maximum(abs.(s.f_vals)) for s in ts_sets),
)

plots = Plots.Plot[]
panel_labels = ["(a)", "(b)", "(c)", "(d)"]
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
let panel_idx = 1
    for (i, s) in enumerate(landau_sets)
        title = latexstring("$(panel_labels[panel_idx]) \$\\epsilon = $(cutoff_label(s.cutoff))\$")
        panel_idx += 1
        is_right = i == 2
        p = heatmap(
            s.x_vals,
            s.v_vals,
            abs.(s.f_vals);
            xlabel = "",
            ylabel = is_right ? "" : L"v",
            yticks = is_right ? nothing : :auto,
            title = title,
            titlelocation = :left,
            colorbar = true,
        )
        push!(plots, p)
    end

    for (i, s) in enumerate(ts_sets)
        title = latexstring("$(panel_labels[panel_idx]) \$\\epsilon = $(cutoff_label(s.cutoff))\$")
        panel_idx += 1
        is_right = i == 2
        p = heatmap(
            s.x_vals,
            s.v_vals,
            abs.(s.f_vals);
            xlabel = L"x",
            ylabel = is_right ? "" : L"v",
            yticks = is_right ? nothing : :auto,
            title = title,
            titlelocation = :left,
            colorbar = true,
        )
        push!(plots, p)
    end
end

plt = plot(
    plots...;
    layout = (2, 2),
    left_margin = 2mm,
    bottom_margin = 2mm,
    size = (833, 650),
)

savefig(plt, "plots/paper_figures/truncation_distr.pdf")
