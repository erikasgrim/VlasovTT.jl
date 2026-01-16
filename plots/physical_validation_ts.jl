using Plots
using LaTeXStrings
using HDF5
using ITensorMPS
using ITensors
using QuanticsGrids
using QuanticsTCI
using VlasovTT: read_data, PhaseSpaceGrids

include(joinpath("plot_defaults.jl"))
PlotDefaults.apply!()

two_stream_ref = "final_results/two_stream/sweep_cutoff/case_004"
two_stream_data = joinpath(two_stream_ref, "data.csv")
two_stream_data = read_data(two_stream_data)

two_stream_xticks = [-1.0, 0, 1.0]
two_stream_yticks = [-0.5, 0, 0.5]

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

# Linear fit
ts_gamma = 0.35355
tmin = 3.0
tmax = 17.0
ts_analytic = 3e-7 .* exp.(2 * ts_gamma .* two_stream_data.times)
linear_indices = findall(i -> (two_stream_data.times[i] >= tmin) && (two_stream_data.times[i] <= tmax), 1:length(two_stream_data.ef_energy))

p1 = plot(
    two_stream_data.times,
    two_stream_data.ef_energy_mode1,
    xlabel = L"t\ [\omega_{pe}^{-1}]",
    ylabel = L"\mathcal{E}_1",
    yaxis = :log10,
    linestyle = :solid,
    label = "Simulation",
    legend = nothing,
    title = "(a)",
    titlelocation = :left,
)
p1 = plot!(
    p1,
    two_stream_data.times[linear_indices],
    ts_analytic[linear_indices],
    linestyle = :dot,
    color = :black,
    label = L"Analytic ($\gamma = 0.3535$)",
)

energy_rel = abs.(two_stream_data.total_energy .- two_stream_data.total_energy[1]) ./ abs(two_stream_data.total_energy[1])
momentum_abs = abs.(two_stream_data.momentum)
positive_min(values) = minimum(filter(>(0), values))
y_limits = (min(positive_min(momentum_abs), positive_min(energy_rel[2:end])),
    max(maximum(momentum_abs), maximum(energy_rel[2:end])))

p2 = plot(
    two_stream_data.times,
    momentum_abs,
    xlabel = L"t\ [\omega_{pe}^{-1}]",
    ylabel = L"|P|",
    ylims = y_limits,
    yaxis = :log10,
    linestyle = :solid,
    color = 2,
    label = "Momentum",
    legend = nothing,
    title = "(b)",
    titlelocation = :left,
)

p3 = plot(
    two_stream_data.times[2:end],
    energy_rel[2:end],
    xlabel = L"t\ [\omega_{pe}^{-1}]",
    ylabel = L"|\Delta E|/|E_0|",
    ylims = y_limits,
    yaxis = :log10,
    linestyle = :solid,
    color =  3,
    label = "Energy",
    legend = nothing,
    title = "(c)",
    titlelocation = :left,
)

println(maximum(energy_rel))

p3_mps_path = joinpath(two_stream_ref, "mps", "psi_step1.h5")
psi_mps = h5open(p3_mps_path, "r") do file
    read(file, "psi", MPS)
end

R, xmin, xmax, vmin, vmax = read_grid_params(joinpath(two_stream_ref, "simulation_params.txt"))
phase = PhaseSpaceGrids(R, xmin, xmax, vmin, vmax)

x_vals = range(phase.xmin, phase.xmax; length = 200)
v_vals = range(phase.vmin, phase.vmax; length = 200)
tt_snapshot = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps))
f_vals = [tt_snapshot(origcoord_to_quantics(phase.x_v_grid, (x, v))) for v in v_vals, x in x_vals]

p4_mps_path = joinpath(two_stream_ref, "mps", "psi_step150.h5")
psi_mps_150 = h5open(p4_mps_path, "r") do file
    read(file, "psi", MPS)
end
tt_snapshot_150 = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps_150))
f_vals_150 = [tt_snapshot_150(origcoord_to_quantics(phase.x_v_grid, (x, v))) for v in v_vals, x in x_vals]

p5_mps_path = joinpath(two_stream_ref, "mps", "psi_step200.h5")
psi_mps_300 = h5open(p5_mps_path, "r") do file
    read(file, "psi", MPS)
end
tt_snapshot_300 = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps_300))
f_vals_300 = [tt_snapshot_300(origcoord_to_quantics(phase.x_v_grid, (x, v))) for v in v_vals, x in x_vals]
clim_max = maximum((maximum(abs.(f_vals)), maximum(abs.(f_vals_150)), maximum(abs.(f_vals_300))))
clim_min = minimum((minimum(abs.(f_vals)), minimum(abs.(f_vals_150)), minimum(abs.(f_vals_300))))
println("Color limits: ", (clim_min, clim_max))

p4 = heatmap(
    x_vals,
    v_vals,
    abs.(f_vals);
    legend = nothing,
    ylabel = L"v",
    title = L"(d)       $t = 0$",
    titlelocation = :left,
    xticks = two_stream_xticks,
    yticks = two_stream_yticks,
    clim = (clim_min, clim_max),
    colorbar = false,
)

p5 = heatmap(
    x_vals,
    v_vals,
    abs.(f_vals_150);
    legend = nothing,
    xlabel = L"x",
    title = L"(e)       $t = 15$",
    titlelocation = :left,
    xticks = two_stream_xticks,
    yticks = nothing,
    clim = (clim_min, clim_max),
    colorbar = false,
)

p6 = heatmap(
    x_vals,
    v_vals,
    abs.(f_vals_300);
    legend = nothing,
    title = L"(f)       $t = 20$               $f(x,v)$",
    titlelocation = :left,
    xticks = two_stream_xticks,
    yticks = nothing,
    clim = (clim_min, clim_max),
    colorbar = :right,
    #colorbar_title = L"f(x,v)",
)

l = @layout [
    grid(1, 1){0.3h}
    grid(1, 2){0.3h}
    grid(1, 3, widths = [0.3, 0.3, 0.4]){0.4h}
]
plt = plot(p1, p2, p3, p4, p5, p6; layout = l, size = (833, 950), left_margin = 2mm)

# Save figure
savefig(plt, "plots/paper_figures/physical_validation_ts.pdf")
