function gaussian_ic(
    phase::PhaseSpaceGrids;
    x0::Real = 0.0,
    v0::Real = 0.0,
    sigma_x::Real = 1.0,
    sigma_v::Real = 1.0,
)
    norm_const = 1 / (2π * sigma_x * sigma_v)
    return quantics -> begin
        coords = quantics_to_origcoord(phase.x_v_grid, quantics)
        x = coords[1]
        v = coords[2]
        return norm_const *
            exp(-((x - x0)^2) / (2 * sigma_x^2)) *
            exp(-((v - v0)^2) / (2 * sigma_v^2))
    end
end

function equilibrium_test_ic(
    phase::PhaseSpaceGrids;
    v0::Real = 0.0,
    sigma_v::Real = 1.0,
)
    norm_const = 1 / (sqrt(2π) * sigma_v)
    return quantics -> begin
        coords = quantics_to_origcoord(phase.x_v_grid, quantics)
        v = coords[2]
        return norm_const * exp(-((v - v0)^2) / (2 * sigma_v^2))
    end
end

function linear_landau_damping_ic(
    phase::PhaseSpaceGrids;
    v0::Real = 0.0,
    sigma_v::Real = 1.0,
    alpha::Real = 0.1,
    mode::Int = 1,
)
    norm_const = 1 / (sqrt(2π) * sigma_v)
    k0 = mode * 2π / phase.Lx
    return quantics -> begin
        coords = quantics_to_origcoord(phase.x_v_grid, quantics)
        x = coords[1]
        v = coords[2]
        return 1 / phase.Lx * (1 + alpha * cos(k0 * x)) * norm_const * exp(-((v - v0)^2) / (2 * sigma_v^2))
    end
end

function two_stream_instability_ic(
    phase::PhaseSpaceGrids;
    A::Real = 0.1,
    v0::Real = 1.0,
    vt::Real = 0.3,
    mode::Int = 3,
)
    k = mode * 2π / phase.Lx
    norm_factor = 1 / (sqrt(2π) * vt)
    return quantics -> begin
        coords = quantics_to_origcoord(phase.x_v_grid, quantics)
        x = coords[1]
        v = coords[2]

        f_beams = 0.5 * norm_factor * (
            exp(-0.5 * ((v - v0) / vt)^2) +
            exp(-0.5 * ((v + v0) / vt)^2)
        )

        return f_beams * (1 + A * cos(k * x))
    end
end

function build_initial_tt(ic_fn, R::Int; tolerance::Real = 1e-8)
    tci, interp_rank, interp_error = TCI.crossinterpolate2(
        Float64,
        ic_fn,
        fill(2, 2R);
        tolerance = tolerance,
    )
    return tci, interp_rank, interp_error
end
