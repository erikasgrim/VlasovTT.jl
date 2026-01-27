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

function min_f_for_steps(ref_path::AbstractString, steps::AbstractVector{Int})
    params_path = joinpath(ref_path, "simulation_params.txt")
    R, xmin, xmax, vmin, vmax = read_grid_params(params_path)
    phase = PhaseSpaceGrids(R, xmin, xmax, vmin, vmax)
    results = Dict{Int,NamedTuple}()
    for step in steps
        mps_path = joinpath(ref_path, "mps", "psi_step$(step).h5")
        isfile(mps_path) || error("Missing MPS file: $(mps_path)")
        psi_mps = h5open(mps_path, "r") do file
            read(file, "psi", MPS)
        end
        tt_snapshot = QuanticsTCI.TensorTrain(ITensors.cpu(psi_mps))
        min_val, min_q, coords, neg_count, npoints = min_f_on_grid(tt_snapshot, phase)
        bitstring = join(string.(min_q .- 1))
        results[step] = (
            min_val = min_val,
            bitstring = bitstring,
            x = coords[1],
            v = coords[2],
            neg_count = neg_count,
            neg_frac = neg_count / npoints,
        )
    end
    return results
end

steps = [1, 150, 300]
landau_ref = "final_results/landau_damping/sweep_cutoff/case_004"


println("Checking Landau damping snapshots...")
landau_results = min_f_for_steps(landau_ref, steps)
for step in steps
    res = landau_results[step]
    println("Landau step $(step) min f: ", res.min_val,
        " | bits: ", res.bitstring,
        " | (x, v): (", res.x, ", ", res.v, ")",
        " | negatives: ", res.neg_count,
        " (", res.neg_frac, ")")
end

steps = [1, 150, 200]
two_stream_ref = "final_results/two_stream/sweep_cutoff/case_004"

println()
println("Checking two-stream snapshots...")
two_stream_results = min_f_for_steps(two_stream_ref, steps)
for step in steps
    res = two_stream_results[step]
    println("Two-stream step $(step) min f: ", res.min_val,
        " | bits: ", res.bitstring,
        " | (x, v): (", res.x, ", ", res.v, ")",
        " | negatives: ", res.neg_count,
        " (", res.neg_frac, ")")
end
