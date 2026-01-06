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

two_stream_ref = "results/two_stream/sweep_cutoff/case_001"
two_stream_data = joinpath(two_stream_ref, "data.csv")
two_stream_data = read_data(two_stream_data)

landau_ref = "results/landau_damping/sweep_cutoff/case_001"
landau_data = joinpath(landau_ref, "data.csv")
landau_data = read_data(landau_data)

landau_xticks = [-5.0, 0, 5.0]
landau_yticks = [-4.5, 0, 4.5]
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

function read_time_step(path)
    for line in eachline(path)
        occursin("dt =", line) || continue
        _, val = strip.(split(line, "=", limit = 2))
        return parse(Float64, val)
    end
    error("dt not found in $(path)")
end


# Linear fit 
landau_gamma = -0.15139
landau_analytic = [1e-1 * exp(2 * landau_gamma * t) for t in landau_data.times]

p1 = plot(
    landau_data.times,
    [landau_data.ef_energy, landau_analytic],
    xlabel = L"t\ [\omega_{pe}^{-1}]",
    ylabel = L"\mathcal{E}_1",
    yaxis = :log10,
    linestyle = [:solid :dot],
    color = [1 :black],
    label = L"Two-Stream Reference",
    legend = nothing
)

ts_gamma = 0.35355
t_min = 3.0
tmax = 15.0
ts_analytic = 3e-7 .* exp.(2 * ts_gamma .* two_stream_data.times)

p2 = plot(
    two_stream_data.times,
    [two_stream_data.ef_energy, ts_analytic],
    xlabel = L"t\ [\omega_{pe}^{-1}]",
    ylabel = L"\mathcal{E}_1",
    yaxis = :log10,
    linestyle = [:solid :dot],
    color = [1 :black],
    label = L"Numerical",
    legend = nothing

)

p3_mps_path = joinpath(landau_ref, "mps", "psi_step1.h5")
psi_mps = h5open(p3_mps_path, "r") do file
    read(file, "psi", MPS)
end

R, xmin, xmax, vmin, vmax = read_grid_params(joinpath(landau_ref, "simulation_params.txt"))
phase = PhaseSpaceGrids(R, xmin, xmax, vmin, vmax)

x_vals = range(phase.xmin, phase.xmax; length = 100)
v_vals = range(phase.vmin, phase.vmax; length = 100)
tt_snapshot = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps))
f_vals = [tt_snapshot(origcoord_to_quantics(phase.x_v_grid, (x, v))) for v in v_vals, x in x_vals]

p3 = heatmap(
    x_vals,
    v_vals,
    abs.(f_vals);
    legend = nothing,
    #xlabel = L"x",
    ylabel = L"v",
    title = L"t = 0",
    xticks = landau_xticks,
    yticks = landau_yticks,
)

p4_mps_path = joinpath(landau_ref, "mps", "psi_step150.h5")
psi_mps_150 = h5open(p4_mps_path, "r") do file
    read(file, "psi", MPS)
end
tt_snapshot_150 = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps_150))
f_vals_150 = [tt_snapshot_150(origcoord_to_quantics(phase.x_v_grid, (x, v))) for v in v_vals, x in x_vals]
p4 = heatmap(
    x_vals,
    v_vals,
    abs.(f_vals_150);
    legend = nothing,
    xlabel = L"x",
    title = L"t = 15",
    xticks = landau_xticks,
    yticks = nothing
)

p5_mps_path = joinpath(landau_ref, "mps", "psi_step300.h5")
psi_mps_300 = h5open(p5_mps_path, "r") do file
    read(file, "psi", MPS)
end
tt_snapshot_300 = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps_300))
f_vals_300 = [tt_snapshot_300(origcoord_to_quantics(phase.x_v_grid, (x, v))) for v in v_vals, x in x_vals]
p5 = heatmap(
    x_vals,
    v_vals,
    abs.(f_vals_300);
    legend = nothing,
    #xlabel = L"x",
    title = L"t = 30",
    xticks = landau_xticks,
    yticks = nothing
)

two_stream_params = joinpath(two_stream_ref, "simulation_params.txt")
R_ts, xmin_ts, xmax_ts, vmin_ts, vmax_ts = read_grid_params(two_stream_params)
dt_ts = read_time_step(two_stream_params)
phase_ts = PhaseSpaceGrids(R_ts, xmin_ts, xmax_ts, vmin_ts, vmax_ts)

x_vals_ts = range(phase_ts.xmin, phase_ts.xmax; length = 100)
v_vals_ts = range(phase_ts.vmin, phase_ts.vmax; length = 100)

p6_mps_path = joinpath(two_stream_ref, "mps", "psi_step1.h5")
psi_mps_ts_1 = h5open(p6_mps_path, "r") do file
    read(file, "psi", MPS)
end
tt_snapshot_ts_1 = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps_ts_1))
f_vals_ts_1 = [tt_snapshot_ts_1(origcoord_to_quantics(phase_ts.x_v_grid, (x, v))) for v in v_vals_ts, x in x_vals_ts]
p6 = heatmap(
    x_vals_ts,
    v_vals_ts,
    abs.(f_vals_ts_1);
    legend = nothing,
    #xlabel = L"x",
    ylabel = L"v",
    title = "t = $(round(1 * dt_ts, digits = 3))",
    xticks = two_stream_xticks,
    yticks = two_stream_yticks,
)

p7_mps_path = joinpath(two_stream_ref, "mps", "psi_step150.h5")
psi_mps_ts_150 = h5open(p7_mps_path, "r") do file
    read(file, "psi", MPS)
end
tt_snapshot_ts_150 = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps_ts_150))
f_vals_ts_150 = [tt_snapshot_ts_150(origcoord_to_quantics(phase_ts.x_v_grid, (x, v))) for v in v_vals_ts, x in x_vals_ts]
p7 = heatmap(
    x_vals_ts,
    v_vals_ts,
    abs.(f_vals_ts_150);
    legend = nothing,
    xlabel = L"x",
    title = "t = $(round(150 * dt_ts, digits = 3))",
    xticks = two_stream_xticks,
    yticks = nothing,
)

p8_mps_path = joinpath(two_stream_ref, "mps", "psi_step250.h5")
psi_mps_ts_300 = h5open(p8_mps_path, "r") do file
    read(file, "psi", MPS)
end
tt_snapshot_ts_300 = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps_ts_300))
f_vals_ts_300 = [tt_snapshot_ts_300(origcoord_to_quantics(phase_ts.x_v_grid, (x, v))) for v in v_vals_ts, x in x_vals_ts]
p8 = heatmap(
    x_vals_ts,
    v_vals_ts,
    abs.(f_vals_ts_300);
    legend = nothing,
    #xlabel = L"x",
    title = "t = $(round(300 * dt_ts, digits = 3))",
    xticks = two_stream_xticks,
    yticks = nothing,
)


l = @layout [grid(1, 2){0.7h}; grid(1, 6){0.3h}]
plt = plot(p1, p2, p3, p4, p5, p6, p7, p8; layout = l)

# Save figure
mkpath("paper_figures")
savefig(plt, "plots/paper_figures/reference_plot.pdf")
