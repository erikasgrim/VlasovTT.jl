using HDF5
using ITensorMPS
using ITensors
using QuanticsGrids
using QuanticsTCI
using VlasovTT: PhaseSpaceGrids

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

function min_f_on_grid(tt_snapshot, phase::PhaseSpaceGrids)
    R = phase.R
    nbits = 2 * R
    if nbits > 62
        error("Grid too large for exhaustive scan: 2R = $(nbits)")
    end
    npoints = 1 << nbits
    q = Vector{Int}(undef, nbits)
    min_val = Inf
    min_q = similar(q)
    neg_count = 0
    for idx in 0:(npoints - 1)
        for b in 1:nbits
            q[b] = ((idx >> (b - 1)) & 1) == 1 ? 2 : 1
        end
        val = real(tt_snapshot(q))
        if val < 0
            neg_count += 1
        end
        if val < min_val
            min_val = val
            copyto!(min_q, q)
        end
    end
    coords = quantics_to_origcoord(phase.x_v_grid, min_q)
    return min_val, min_q, coords, neg_count, npoints
end

function min_f_for_case(case_dir::AbstractString, step::Int)
    params_path = joinpath(case_dir, "simulation_params.txt")
    R, xmin, xmax, vmin, vmax = read_grid_params(params_path)
    phase = PhaseSpaceGrids(R, xmin, xmax, vmin, vmax)

    mps_path = joinpath(case_dir, "mps", "psi_step$(step).h5")
    isfile(mps_path) || error("Missing MPS file: $(mps_path)")
    psi_mps = h5open(mps_path, "r") do file
        read(file, "psi", MPS)
    end
    tt_snapshot = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps))
    min_val, min_q, coords, neg_count, npoints = min_f_on_grid(tt_snapshot, phase)
    bitstring = join(string.(min_q .- 1))
    return (
        min_val = min_val,
        bitstring = bitstring,
        x = coords[1],
        v = coords[2],
        neg_count = neg_count,
        neg_frac = neg_count / npoints,
    )
end

function summarize_case(label::String, cutoff::Float64, res)
    println(label, " cutoff: ", cutoff)
    println("  min f: ", res.min_val,
        " | bits: ", res.bitstring,
        " | (x, v): (", res.x, ", ", res.v, ")",
        " | negatives: ", res.neg_count,
        " (", res.neg_frac, ")")
end

landau_dir = "final_results/landau_damping/sweep_cutoff"
ts_dir = "final_results/two_stream/sweep_cutoff"
step = 300

landau_sets = load_cases(landau_dir)
ts_sets = load_cases(ts_dir)
if length(landau_sets) >= 2
    landau_sets = [last(landau_sets), first(landau_sets)]
end
if length(ts_sets) >= 2
    ts_sets = [last(ts_sets), first(ts_sets)]
end

println("Checking truncation_distr snapshots (step $(step))...")
for (i, s) in enumerate(landau_sets)
    res = min_f_for_case(s.path, step)
    summarize_case("Landau $(i)", s.cutoff, res)
end
for (i, s) in enumerate(ts_sets)
    res = min_f_for_case(s.path, step)
    summarize_case("Two-stream $(i)", s.cutoff, res)
end
