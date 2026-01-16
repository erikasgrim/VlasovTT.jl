struct ObservablesCache
    ones_mps::MPS
    v2_mps::MPS
    v_mps
end

function build_observables_cache(psi_mps::MPS, phase::PhaseSpaceGrids; tolerance::Real = 1e-12)
    sites = siteinds(psi_mps)

    v2_fn = quantics -> begin
        coords = quantics_to_origcoord_xv(phase, quantics)
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
        coords = quantics_to_origcoord_xv(phase, quantics)
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

function electric_field_mode_energy(
    electric_field_mps::MPS,
    phase::PhaseSpaceGrids,
    fourier_mpo_it::MPO;
    mode::Int = 1,
)
    electric_field_hat = apply(fourier_mpo_it, electric_field_mps; alg = "naive")

    electric_field_hat_tt = TCI.TensorTrain(ITensors.cpu(electric_field_hat))
    k_pos = n_to_k(mode, phase.M)
    k_neg = n_to_k(-mode, phase.M)
    qk_pos = maybe_reverse_bits(origcoord_to_quantics(phase.kx_grid, k_pos), phase.kx_lsb_first)
    qk_neg = maybe_reverse_bits(origcoord_to_quantics(phase.kx_grid, k_neg), phase.kx_lsb_first)
    energy = abs2(electric_field_hat_tt(qk_pos)) + abs2(electric_field_hat_tt(qk_neg))

    return 0.5 * phase.dx * energy
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

function total_charge_kv(psi_mps::MPS, phase::PhaseSpaceGrids)
    # Project v sites to k_v = 0, then integrate over x.
    cd_mps = get_charge_density_kv(
        psi_mps;
        dv=sqrt(phase.M) * phase.dv,
        unfoldingscheme=phase.unfoldingscheme,
    )  # x-only MPS
    ones_x = ones_mps(siteinds(cd_mps))
    charge = inner(ones_x, ITensors.cpu(cd_mps))
    return charge * phase.dx
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
