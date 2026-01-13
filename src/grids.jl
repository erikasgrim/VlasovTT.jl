struct PhaseSpaceGrids
    R::Int
    M::Int
    xmin::Real
    xmax::Real
    vmin::Real
    vmax::Real
    Lx::Real
    Lv::Real
    dx::Real
    dv::Real
    x_grid
    v_grid
    x_v_grid
    kx_grid
    kv_grid
    x_lsb_first::Bool
    v_lsb_first::Bool
    kx_lsb_first::Bool
    kv_lsb_first::Bool
end

function PhaseSpaceGrids(
    R::Int,
    xmin::Real,
    xmax::Real,
    vmin::Real,
    vmax::Real;
    unfoldingscheme=:interleaved,
    x_lsb_first::Bool=true,
    v_lsb_first::Bool=false,
)
    M = 2^R
    Lx = xmax - xmin
    Lv = vmax - vmin
    dx = Lx / M
    dv = Lv / M

    x_grid = DiscretizedGrid{1}(R, xmin, xmax)
    v_grid = DiscretizedGrid{1}(R, vmin, vmax)
    x_v_grid = DiscretizedGrid{2}(R, (xmin, vmin), (xmax, vmax); unfoldingscheme=unfoldingscheme)
    kx_grid = InherentDiscreteGrid{1}(R, 0)
    kv_grid = InherentDiscreteGrid{1}(R, 0)

    kx_lsb_first = !x_lsb_first
    kv_lsb_first = !v_lsb_first

    return PhaseSpaceGrids(
        R,
        M,
        xmin,
        xmax,
        vmin,
        vmax,
        Lx,
        Lv,
        dx,
        dv,
        x_grid,
        v_grid,
        x_v_grid,
        kx_grid,
        kv_grid,
        x_lsb_first,
        v_lsb_first,
        kx_lsb_first,
        kv_lsb_first,
    )
end

function quantics_to_origcoord_xv(phase::PhaseSpaceGrids, q_bits::AbstractVector{Int})
    q_x, q_v = split_interleaved_bits(q_bits, phase.R)
    q_x_aligned = maybe_reverse_bits(q_x, phase.x_lsb_first)
    q_v_aligned = maybe_reverse_bits(q_v, phase.v_lsb_first)
    x = quantics_to_origcoord(phase.x_grid, q_x_aligned)
    v = quantics_to_origcoord(phase.v_grid, q_v_aligned)
    return (x, v)
end

function origcoord_to_quantics_xv(phase::PhaseSpaceGrids, x::Real, v::Real)
    q_x = origcoord_to_quantics(phase.x_grid, x)
    q_v = origcoord_to_quantics(phase.v_grid, v)
    q_x = maybe_reverse_bits(q_x, phase.x_lsb_first)
    q_v = maybe_reverse_bits(q_v, phase.v_lsb_first)
    return interleave_bits(q_x, q_v)
end
