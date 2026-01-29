using Plots
using Plots.PlotMeasures
using LaTeXStrings
using HDF5
using ITensorMPS
using ITensors
using QuanticsGrids
using QuanticsTCI
using VlasovTT: read_data, PhaseSpaceGrids

include(joinpath("plot_defaults.jl"))
PlotDefaults.apply!()

landau_ref = "final_results/landau_damping/sweep_cutoff/case_004"
landau_data = joinpath(landau_ref, "data.csv")
landau_data = read_data(landau_data)

landau_xticks = [-5.0, 0, 5.0]
landau_yticks = [-4.5, 0, 4.5]

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
landau_gamma = -0.15139
landau_analytic = [1e-1 * exp(2 * landau_gamma * t) for t in landau_data.times]

p1 = plot(
    landau_data.times,
    [landau_data.ef_energy_mode1, landau_analytic],
    xlabel = L"t\ [\omega_{pe}^{-1}]",
    ylabel = L"\mathcal{E}_1",
    yaxis = :log10,
    linestyle = [:solid :dot],
    color = [1 :black],
    label = L"Analytic ($\gamma = 0.1514)$",
    legend = nothing,
    title = "(a)",
    titlelocation = :left,
)

energy_rel = abs.(landau_data.total_energy .- landau_data.total_energy[1]) ./ abs(landau_data.total_energy[1])
momentum_abs = abs.(landau_data.momentum)
positive_min(values) = minimum(filter(>(0), values))
y_limits = (min(positive_min(momentum_abs), positive_min(energy_rel[2:end])),
    max(maximum(momentum_abs), maximum(energy_rel[2:end])))

p2 = plot(
    landau_data.times,
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
    landau_data.times[2:end],
    energy_rel[2:end],
    xlabel = L"t\ [\omega_{pe}^{-1}]",
    ylabel = L"|E - E_0|/|E_0|",
    ylims = y_limits,
    yaxis = :log10,
    linestyle = :solid,
    color =  3,
    label = "Energy",
    legend = nothing,
    title = "(c)",
    titlelocation = :left,
)

p3_mps_path = joinpath(landau_ref, "mps", "psi_step1.h5")
psi_mps = h5open(p3_mps_path, "r") do file
    read(file, "psi", MPS)
end

R, xmin, xmax, vmin, vmax = read_grid_params(joinpath(landau_ref, "simulation_params.txt"))
phase = PhaseSpaceGrids(R, xmin, xmax, vmin, vmax)

x_vals = range(phase.xmin, phase.xmax; length = 200)
v_vals = range(phase.vmin, phase.vmax; length = 200)
tt_snapshot = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps))
f_vals = [tt_snapshot(origcoord_to_quantics(phase.x_v_grid, (x, v))) for v in v_vals, x in x_vals]
f_vals_real = real.(f_vals)
f_vals_plot = ifelse.(f_vals_real .< 0, NaN, f_vals_real)

p4_mps_path = joinpath(landau_ref, "mps", "psi_step150.h5")
psi_mps_150 = h5open(p4_mps_path, "r") do file
    read(file, "psi", MPS)
end
tt_snapshot_150 = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps_150))
f_vals_150 = [tt_snapshot_150(origcoord_to_quantics(phase.x_v_grid, (x, v))) for v in v_vals, x in x_vals]
f_vals_150_real = real.(f_vals_150)
f_vals_150_plot = ifelse.(f_vals_150_real .< 0, NaN, f_vals_150_real)

p5_mps_path = joinpath(landau_ref, "mps", "psi_step300.h5")
psi_mps_300 = h5open(p5_mps_path, "r") do file
    read(file, "psi", MPS)
end
tt_snapshot_300 = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps_300))
f_vals_300 = [tt_snapshot_300(origcoord_to_quantics(phase.x_v_grid, (x, v))) for v in v_vals, x in x_vals]
f_vals_300_real = real.(f_vals_300)
f_vals_300_plot = ifelse.(f_vals_300_real .< 0, NaN, f_vals_300_real)
clim_max = max(0.0, maximum((maximum(f_vals_real), maximum(f_vals_150_real), maximum(f_vals_300_real))))
println("Color limits: ", (0.0, clim_max))

p4 = heatmap(
    x_vals,
    v_vals,
    f_vals_plot;
    legend = nothing,
    ylabel = L"v",
    title = L"(d)       $t = 0$",
    titlelocation = :left,
    xticks = landau_xticks,
    yticks = landau_yticks,
    clim = (0.0, clim_max),
    c = :viridis,
    nan_color = :white,
    colorbar = false,
)

p5 = heatmap(
    x_vals,
    v_vals,
    f_vals_150_plot;
    legend = nothing,
    xlabel = L"x",
    title = L"(e)       $t = 15$",
    titlelocation = :left,
    xticks = landau_xticks,
    yticks = nothing,
    clim = (0.0, clim_max),
    c = :viridis,
    nan_color = :white,
    colorbar = false,
)

p6 = heatmap(
    x_vals,
    v_vals,
    f_vals_300_plot;
    legend = nothing,
    title = L"(f)       $t = 30$               $f(x,v)$",
    titlelocation = :left,
    xticks = landau_xticks,
    yticks = nothing,
    clim = (0.0, clim_max),
    c = :viridis,
    nan_color = :white,
    colorbar = :right,
    #colorbar_title = L"f(x,v)",
)

l = @layout [
    grid(1, 1){0.3h}
    grid(1, 2){0.3h}
    grid(1, 3, widths = [0.29, 0.29, 0.42]){0.4h}
]
plt = plot(p1, p2, p3, p4, p5, p6; layout = l, size = (833, 950), left_margin = 2mm)

# Save figure
savefig(plt, "plots/paper_figures/physical_validation_landau_mask.pdf")
