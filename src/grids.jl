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
end

function PhaseSpaceGrids(R::Int, xmin::Real, xmax::Real, vmin::Real, vmax::Real; unfoldingscheme=:interleaved)
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

    return PhaseSpaceGrids(R, M, xmin, xmax, vmin, vmax, Lx, Lv, dx, dv, x_grid, v_grid, x_v_grid, kx_grid, kv_grid)
end
