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
landau_analytic = [2e-1 * exp(2 * landau_gamma * t) for t in landau_data.times]

p1 = plot(
    landau_data.times,
    [landau_data.ef_energy, landau_analytic],
    xlabel = L"t\ [\omega_{pe}^{-1}]",
    ylabel = L"\mathcal{E}",
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

p2 = plot(
    landau_data.times,
    momentum_abs,
    xlabel = L"t\ [\omega_{pe}^{-1}]",
    ylabel = L"|P|",
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
    ylabel = L"|\Delta E|/|E_0|",
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

p4_mps_path = joinpath(landau_ref, "mps", "psi_step150.h5")
psi_mps_150 = h5open(p4_mps_path, "r") do file
    read(file, "psi", MPS)
end
tt_snapshot_150 = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps_150))
f_vals_150 = [tt_snapshot_150(origcoord_to_quantics(phase.x_v_grid, (x, v))) for v in v_vals, x in x_vals]

p5_mps_path = joinpath(landau_ref, "mps", "psi_step300.h5")
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
    xticks = landau_xticks,
    yticks = landau_yticks,
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
    xticks = landau_xticks,
    yticks = nothing,
    clim = (clim_min, clim_max),
    colorbar = false,
)

p6 = heatmap(
    x_vals,
    v_vals,
    abs.(f_vals_300);
    legend = nothing,
    title = L"(f)       $t = 30$               $f(x,v)$",
    titlelocation = :left,
    xticks = landau_xticks,
    yticks = nothing,
    clim = (clim_min, clim_max),
    colorbar = :right,
    #colorbar_title = L"f(x,v)",
)

l = @layout [
    grid(1, 1){0.4h}
    grid(1, 2){0.3h}
    grid(1, 3, widths = [0.3, 0.3, 0.4]){0.3h}
]
plt = plot(p1, p2, p3, p4, p5, p6; layout = l, size = (833, 750))

# Save figure
savefig(plt, "plots/paper_figures/physical_validation_landau.pdf")
