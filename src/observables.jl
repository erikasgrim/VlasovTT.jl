struct ObservablesCache
    ones_mps::MPS
    v2_mps::MPS
    v_mps
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
    println("(Observable) v^2  TT ranks: ", TCI.rank(v2_tt))
    v2_mps = MPS(v2_tt; sites = sites)

    v_fn = quantics -> begin
        coords = quantics_to_origcoord(phase.x_v_grid, quantics)
        v = coords[2]
        return v
    end

    v_tt, _, _ = TCI.crossinterpolate2(
        Float64,
        v_fn,
        fill(2, 2 * phase.R);
        tolerance = tolerance,
    )
    println("(Observable) v TT ranks: ", TCI.rank(v_tt))
    v_mps = MPS(v_tt; sites = sites)

    return ObservablesCache(ones_mps(sites), v2_mps, v_mps)
end

function electric_field_energy(electric_field_mps::MPS, phase::PhaseSpaceGrids)
    # ∑ |E(x)|^2 dx
    energy_density = inner(electric_field_mps, electric_field_mps)
    return 0.5 * energy_density * phase.dx
end

function total_charge(psi_mps::MPS, phase::PhaseSpaceGrids, cache::Union{ObservablesCache,Nothing})
    # ∑ f(x,v) dx dv
    ones_state = cache.ones_mps
    charge = inner(ones_state, ITensors.cpu(psi_mps))
    return charge * phase.dx * phase.dv
end

function kinetic_energy(psi_mps::MPS, phase::PhaseSpaceGrids, cache::Union{ObservablesCache,Nothing})
    # 0.5 ∑ v^2 f(x,v) dx dv
    f_mps_cpu = ITensors.cpu(psi_mps)
    ke_density = inner(f_mps_cpu, cache.v2_mps)
    return 0.5 * ke_density * phase.dx * phase.dv
end

function total_momentum(psi_mps::MPS, phase::PhaseSpaceGrids, cache::Union{ObservablesCache,Nothing})
    # ∑ v f(x,v) dx dv
    f_mps_cpu = ITensors.cpu(psi_mps)
    momentum_density = inner(f_mps_cpu, cache.v_mps)
    return momentum_density * phase.dx * phase.dv
end
