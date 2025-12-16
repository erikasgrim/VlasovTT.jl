struct ObservablesCache
    ones_mps::MPS
    v2_mps::MPS
end

function build_observables_cache(psi_mps::MPS, phase::PhaseSpaceGrids; tolerance::Real = 1e-12)
    sites = siteinds(psi_mps)

    v2_fn = quantics -> begin
        coords = quantics_to_origcoord(phase.x_v_grid, quantics)
        v = coords[2]
        return v^2
    end

    v2_tt, _, _ = TCI.crossinterpolate2(
        Float64,
        v2_fn,
        fill(2, 2 * phase.R);
        tolerance = tolerance,
    )
    v2_mps = MPS(v2_tt; sites = sites)

    return ObservablesCache(ones_mps(sites), v2_mps)
end

function electric_field_energy(electric_field_mps::MPS, phase::PhaseSpaceGrids)
    # Discrete ∑ |E(x)|^2 dx, E is only on x-bits
    energy_density = inner(electric_field_mps, electric_field_mps)
    return 0.5 * energy_density * phase.dx
end

function total_charge(psi_mps::MPS, phase::PhaseSpaceGrids; cache::Union{ObservablesCache,Nothing} = nothing)
    # Discrete ∑ f(x,v) dx dv
    ones_state = isnothing(cache) ? ones_mps(siteinds(psi_mps)) : cache.ones_mps
    charge = inner(ones_state, ITensors.cpu(psi_mps))
    return charge * phase.dx * phase.dv
end

function kinetic_energy(psi_mps::MPS, phase::PhaseSpaceGrids; cache::Union{ObservablesCache,Nothing} = nothing, tolerance::Real = 1e-8)
    # 0.5 ∑ v^2 f(x,v) dx dv
    cache = isnothing(cache) ? build_observables_cache(psi_mps, phase; tolerance = tolerance) : cache

    f_mps_cpu = ITensors.cpu(psi_mps)
    ke_density = inner(f_mps_cpu, cache.v2_mps)
    return 0.5 * ke_density * phase.dx * phase.dv
end
