using LinearAlgebra

import TensorCrossInterpolation as TCI
using QuanticsGrids
using QuanticsTCI

using ITensorMPS
using ITensors
using CUDA

using Plots
using LaTeXStrings

using ProgressBars

include("src/utilities.jl")
include("src/fourier.jl")

function interleave_bits(q_kx::AbstractVector{Int}, q_v::AbstractVector{Int})
    R = length(q_kx)
    @assert length(q_v) == R
    q = Vector{Int}(undef, 2R)
    for r in 1:R
        q[2r-1] = q_kx[r]  # odd positions: kx
        q[2r]   = q_v[r]   # even positions: v
    end
    return q
end

function Theta(n; beta::Real = 2.0, k_cut::Real = 2^8)
    return 1 / (exp((abs(n) - k_cut) * beta) + 1)
end

function free_streaming_pivots(kx_grid, v_grid, Mx; lsb_first::Bool = false)
    R = length(kx_grid)

    # Representative kx modes (inherent coordinates)
    ks = [0, 2, Mx - 2, Mx - 1]

    # Representative v positions (continuous)
    vmin = quantics_to_origcoord(v_grid, fill(1, R))
    vmax = quantics_to_origcoord(v_grid, fill(2, R))
    vmid = 0.5 * (vmin + vmax)
    vs = [vmin, vmid, vmax]

    pivots = Vector{Vector{Int}}()

    for k in ks
        q_kx = origcoord_to_quantics(kx_grid, k)
        q_kx = lsb_first ? reverse(q_kx) : q_kx
        for v in vs
            q_v = origcoord_to_quantics(v_grid, v)
            push!(pivots, interleave_bits(q_kx, q_v))
        end
    end

    return pivots
end

function get_free_streaming_mpo(
    dt::Real,
    Lx::Real,
    Mx::Int,                  # number of x grid points (2^R)
    kx_grid,                  # InherentDiscreteGrid{1}(R, 0) for kx
    v_grid;                   # DiscretizedGrid{1}(R, vmin, vmax) for v
    tolerance::Real = 1e-12,
    k_cut::Real = 2^8,
    beta::Real = 2.0,
    lsb_first::Bool = false,
)
    # Interleaved layout: (kx₁, v₁, kx₂, v₂, ..., kx_R, v_R)
    # So we have 2R physical bits, all with local dim 2
    R = length(kx_grid)
    localdims = fill(2, 2R)  # 2R bits

    # Build representative pivots
    initial_pivots = free_streaming_pivots(kx_grid, v_grid, Mx; lsb_first = lsb_first)

    # Kernel in *quantics coordinate* space
    function kernel(q_bits::AbstractVector{Int})
        # q_bits is length 2R: [kx₁, v₁, kx₂, v₂, …]

        # Extract kx and v bits (odd indices → kx, even → v)
        q_kx = q_bits[1:2:2R]   # 1,3,5,…,2R-1
        q_v  = q_bits[2:2:2R]   # 2,4,6,…,2R
        q_kx_aligned = lsb_first ? reverse(q_kx) : q_kx

        # Map quantics → original coordinates
        kx_orig = quantics_to_origcoord(kx_grid, q_kx_aligned)  # inherent discrete coord
        v_orig  = quantics_to_origcoord(v_grid,  q_v)   # physical velocity

        # Convert discrete kx index → Fourier mode n_x → physical kₓ
        n_x = k_to_n(kx_orig, Mx)          # n_x ∈ [-Mx/2+1, …, Mx/2]
        kx_phys = 2π * n_x / Lx           # physical k_x

        return exp(-1im * kx_phys * v_orig * dt) * Theta(n_x; beta = beta, k_cut = k_cut)
        #return 1 - 1im * kx_phys * v_orig * dt - 0.5 * (kx_phys^2) * (v_orig^2) * (dt^2)
    end

    # Build QTT for U_A(kx, v) using crossinterpolate2 over interleaved bits
    tci, _, _ = TCI.crossinterpolate2(
        ComplexF64, 
        kernel, 
        localdims, 
        initial_pivots; 
        tolerance = tolerance
    )
    U_tt = TCI.TensorTrain(tci)
    println("Free streaming MPO ranks: ", rank(U_tt))

    # Convert TT → MPO
    return tt_to_mpo(U_tt)
end

function acceleration_pivots(x_grid, kv_grid, Mv; lsb_first::Bool = false)
    R = length(x_grid)

    # Representative x positions (continuous)
    xmin = quantics_to_origcoord(x_grid, fill(1, R))
    xmax = quantics_to_origcoord(x_grid, fill(2, R))
    xmid = 0.5 * (xmin + xmax)
    xs = [xmin, xmid, xmax]

    # Representative kv modes (integer indices 0..Mv-1 in inherent grid)
    ks = [0, 2, Mv - 2, Mv - 1]

    pivots = Vector{Vector{Int}}()

    for x in xs
        q_x = origcoord_to_quantics(x_grid, x)
        for k in ks
            q_kv = origcoord_to_quantics(kv_grid, k)
            q_kv = lsb_first ? reverse(q_kv) : q_kv
            push!(pivots, interleave_bits(q_x, q_kv))
        end
    end

    return pivots
end

function get_acceleration_mpo(
    dt::Real,
    Lv::Real,                      # length of v-domain (vmax - vmin)
    Mv::Int,                       # number of v grid points (2^R)
    x_grid,                        # DiscretizedGrid{1}(R, xmin, xmax)
    kv_grid,                       # InherentDiscreteGrid{1}(R, 0) for kv
    electric_field_tt::TCI.TensorTrain;
    tolerance::Real = 1e-12,
    k_cut::Real = 2^8,
    beta::Real = 2.0,
    lsb_first::Bool = false,
)
    # Interleaved layout: (x₁, kv₁, x₂, kv₂, ..., x_R, kv_R)
    R = length(x_grid)
    @assert length(kv_grid) == R
    localdims = fill(2, 2R)

    # Build representative pivots
    initial_pivots = acceleration_pivots(x_grid, kv_grid, Mv; lsb_first = lsb_first)

    # Kernel in quantics coordinate space for (x, kv)
    function kernel(q_bits::AbstractVector{Int})
        # q_bits is length 2R: [x₁, kv₁, x₂, kv₂, …]

        # Extract x and kv bits (odd indices → x, even → kv)
        q_x  = q_bits[1:2:2R]
        q_kv = q_bits[2:2:2R]

        # Align x bits if using LSB-first convention
        q_kv_aligned = lsb_first ? reverse(q_kv) : q_kv

        # Map kv quantics → inherent discrete coordinate (0...Mv-1)
        kv_orig = quantics_to_origcoord(kv_grid, q_kv_aligned)
        n_v     = k_to_n(kv_orig, Mv)        # n_v ∈ [-Mv/2+1, …, Mv/2]
        kv_phys = 2π * n_v / Lv              # physical k_v

        # Evaluate E(x) directly in quantics coordinates
        E_val = electric_field_tt(q_x)

        phase = -E_val * kv_phys * dt

        # Optional spectral filter in kv-space
        #return exp(-1im * E_val * kv_phys * dt) * Theta(n_v; beta = beta, k_cut = k_cut)
        return ( 1 + 1im * phase - 0.5 * phase^2 ) #* Theta(n_v; beta = beta, k_cut = k_cut)
    end

    # Build QTT for U_E(x, kv) using crossinterpolate2 over interleaved bits
    tci, _, _ = TCI.crossinterpolate2(
        ComplexF64,
        kernel,
        localdims,
        initial_pivots;
        tolerance = tolerance,
    )
    U_tt = TCI.TensorTrain(tci)
    #println("Acceleration MPO ranks: ", rank(U_tt))

    # Convert TT → MPO
    return tt_to_mpo(U_tt)
end

function get_poisson_mpo(
    Lx::Real,
    Mx::Int,
    kx_grid;                # 1D grid over kx (InherentDiscreteGrid{1})
    tolerance::Real = 1e-12,
    eps0::Real = 1.0        # Include permittivity if you want
)
    poisson_kernel = kx -> begin
        nx = k_to_n(kx, Mx)
        if nx == 0
            return 0.0 + 0.0im  # Enforce zero mean field for k=0
        else
            kx_phys = 2π * nx / Lx
            return (-1im / (eps0 * kx_phys))  # \hat E = (i/k) \hat rho / eps0
        end
    end

    qtci, _, _ = quanticscrossinterpolate(
        ComplexF64,
        poisson_kernel,
        kx_grid;
        tolerance = tolerance,
    )
    tt = TCI.TensorTrain(qtci.tci)
    println("Poisson MPO ranks: ", rank(tt))
    return tt_to_mpo(tt)
end

function get_charge_density(psi::MPS; dv::Real=1.0)
    N = length(psi)
    N == 0 && return psi

    # Work on CPU to avoid mixed CPU/GPU contractions when psi lives on a GPU.
    working = copy(ITensors.cpu(psi))
    for i in 2:2:N
        si = siteind(working, i)
        ones_vec = ITensor(si)
        for n in 1:dim(si)
            ones_vec[si => n] = 1.0
        end

        Ai = working[i] * ones_vec
        if i < N
            working[i + 1] = Ai * working[i + 1]
        else
            working[i - 1] = working[i - 1] * Ai
        end
        working[i] = ITensor(1.0)
    end

    odd_tensors = [working[i] for i in 1:N if isodd(i)]
    odd_tensors[1] *= dv
    return MPS(odd_tensors)
end

function ones_mps(sites)
    N = length(sites)
    N == 0 && return MPS()

    # Build dim-1 link indices so bonds are well-defined for TT/MPS conversions.
    links = N > 1 ? [Index(1, "Link,l=$i") for i in 1:(N - 1)] : Index[]
    tensors = Vector{ITensor}(undef, N)

    for i in 1:N
        si = sites[i]

        inds_i = if N == 1
            (si,)
        elseif i == 1
            (links[i], si)
        elseif i == N
            (links[i - 1], si)
        else
            (links[i - 1], links[i], si)
        end

        Ti = ITensor(inds_i...)
        for n in 1:dim(si)
            if N == 1
                Ti[si => n] = 1.0
            elseif i == 1
                Ti[links[i] => 1, si => n] = 1.0
            elseif i == N
                Ti[links[i - 1] => 1, si => n] = 1.0
            else
                Ti[links[i - 1] => 1, links[i] => 1, si => n] = 1.0
            end
        end

        tensors[i] = Ti
    end

    return MPS(tensors)
end

function get_electric_field_mps(
    psi_mps::MPS,
    full_poisson_mpo::TCI.TensorTrain;
    dv::Real=1.0,
    tolerance=1e-8,
    maxrank=nothing,
    alg="naive",
)
    cd_mps = get_charge_density(psi_mps; dv)

    id_mps = ones_mps(siteinds(cd_mps))
    cd_mps = id_mps - cd_mps        # Electric field depends on (1 - ρ)

    sites_cd_mps = siteinds(cd_mps)
    sites_poisson_mpo = [[prime(s,1), s] for s in sites_cd_mps]
    full_poisson_mpo_it = MPO(full_poisson_mpo; sites=sites_poisson_mpo)
    electric_field_mps = apply(full_poisson_mpo_it, cd_mps; alg=alg, truncate=true, maxdim=maxrank, cutoff=tolerance)

    return electric_field_mps
end

let
    # Time evolution parameters
    dt = 1e-2
    T = 10.0
    nsteps = Int(T / dt)

    # Spatial and velocity grids
    R = 10
    M = 2^R

    xmin = -10.0
    xmax = 10.0
    Lx = xmax - xmin
    dx = Lx / M

    vmin = -6.0
    vmax = 6.0
    Lv = vmax - vmin
    dv = Lv / M

    # TCI parameters
    tolerance = 1e-8
    maxrank = 128

    # Frequency filter
    k_cut = 2^6
    beta = 2.0

    # Define grids
    x_v_grid = DiscretizedGrid{2}(R, (xmin, vmin), (xmax, vmax); unfoldingscheme=:interleaved)

    x_grid = DiscretizedGrid{1}(R, xmin, xmax)
    v_grid = DiscretizedGrid{1}(R, vmin, vmax)
    kx_grid = InherentDiscreteGrid{1}(R, 0)   # integer kx index 0...M-1
    kv_grid = InherentDiscreteGrid{1}(R, 0)   # integer kv index 0...M-1

    # Fourier transform MPOs
    x_fourier_mpo = stretched_fourier_mpo(R, 1, 2; sign=-1.0, tolerance=tolerance)
    v_fourier_mpo = stretched_fourier_mpo(R, 2, 2; sign=-1.0, tolerance=tolerance)
    x_inv_fourier_mpo = stretched_fourier_mpo(R, 1, 2; sign=1.0, tolerance=tolerance, lsb_first=true)
    v_inv_fourier_mpo = stretched_fourier_mpo(R, 2, 2; sign=1.0, tolerance=tolerance, lsb_first=true)

    fourier_mpo = quanticsfouriermpo(R; sign=-1.0, tolerance=tolerance)
    inv_fourier_mpo = quanticsfouriermpo(R; sign=1.0, tolerance=tolerance)

    # Free streaming MPO
    free_streaming_mpo = get_free_streaming_mpo(dt, Lx, M, kx_grid, v_grid;
        tolerance = tolerance,
        k_cut = k_cut,
        beta = beta,
        lsb_first = true,
    )

    full_free_streaming_mpo = TCI.contract(
        x_inv_fourier_mpo,
        TCI.contract(
            free_streaming_mpo,
            x_fourier_mpo;
            algorithm=:naive,
            tolerance=tolerance,
        );
        algorithm=:naive,
        tolerance=tolerance,
    )
    println("Full free streaming MPO ranks: ", rank(full_free_streaming_mpo))

    # Poisson / Electric field MPO
    poisson_mpo = get_poisson_mpo(Lx, M, kx_grid;
        tolerance = tolerance,
        eps0 = 1.0,
    )

    full_poisson_mpo = TCI.contract(
        reverse(inv_fourier_mpo),
        TCI.contract(
            reverse(poisson_mpo),
            fourier_mpo;
            algorithm=:naive,
            tolerance=tolerance,
        );
        algorithm=:naive,
        tolerance=tolerance,
    )
    println("Full Poisson MPO ranks: ", rank(full_poisson_mpo))

    # Initial condition: Gaussian in x and v
    x0 = 0.0
    v0 = 0.0
    sigma_x = 3.0
    sigma_v = 1.0
    norm_const = 1 / (2π * sigma_x * sigma_v)
    function initial_condition(quantics)
        coords = quantics_to_origcoord(x_v_grid, quantics)
        x = coords[1]
        v = coords[2]
        return norm_const *
            exp(-((x - x0)^2) / (2 * sigma_x^2)) *
            exp(-((v - v0)^2) / (2 * sigma_v^2))
    end

    function equilibrium_test(quantics)
        coords = quantics_to_origcoord(x_v_grid, quantics)
        x = coords[1]
        v = coords[2]
        norm_const = 1 / (sqrt(2π) * sigma_v)
        return norm_const * exp(-((v - v0)^2) / (2 * sigma_v^2))
    end

    function linear_landau_damping(quantics)
        coords = quantics_to_origcoord(x_v_grid, quantics)
        x = coords[1]
        v = coords[2]
        k0 = 2pi / Lx
        alpha = 0.1
        norm_const = 1 / (sqrt(2π) * sigma_v)
        return 1 / Lx * (1 + alpha * cos(k0 * x)) * norm_const * exp(-((v - v0)^2) / (2 * sigma_v^2))
    end

    function two_stream_instability(quantics)
        coords = quantics_to_origcoord(x_v_grid, quantics)
        x = coords[1]
        v = coords[2]

        A  = 0.1
        v0 = 1.0
        vt = 0.3
        k  = 3 * 2π / Lx   # Lx = xmax - xmin
        norm_factor = 1 / (sqrt(2π) * vt)

        # Two drifting Maxwellian beams
        f_beams = 0.5 * norm_factor * (
            exp(-0.5 * ((v - v0) / vt)^2) +
            exp(-0.5 * ((v + v0) / vt)^2)
        )

        # Small density modulation in x to seed the two-stream mode
        return f_beams * (1 + A * cos(k * x))
    end

    tci, interp_rank, interp_error = TCI.crossinterpolate2(
        Float64,
        two_stream_instability,
        fill(2, 2R);
        tolerance = tolerance,
    )
    tt = TCI.TensorTrain(tci)
    println("Initial condition TT ranks: ", rank(tt))

    # Convert to MPS for ITensors
    psi_mps = MPS(tt)
    sites_mps = siteinds(psi_mps)
    sites_mpo = [[prime(s,1), s] for s in sites_mps]

    # Free stream MPO
    free_stream_mpo_it = MPO(full_free_streaming_mpo; sites=sites_mpo)
    v_fourier_mpo_it = MPO(v_fourier_mpo; sites=sites_mpo)
    v_inv_fourier_mpo_it = MPO(v_inv_fourier_mpo; sites=sites_mpo)

    gpu = true
    if gpu
        psi_mps = cu(psi_mps)
        free_stream_mpo_it = cu(free_stream_mpo_it)
        v_fourier_mpo_it = cu(v_fourier_mpo_it)
        v_inv_fourier_mpo_it = cu(v_inv_fourier_mpo_it)
    end

    # For plotting
    x_vals = range(xmin, xmax; length=200)
    v_vals = range(vmin, vmax; length=200)

    for step in ProgressBar(1:nsteps)
        #tt = TCI.contract(full_free_streaming_mpo, tt; algorithm=:naive, tolerance=tolerance, maxbonddim=maxrank)
        #println("After step $step, TT ranks: ", rank(tt))

        println("Step $step")

        # 1) --- first half acceleration (dt/2) ---
        electric_field_mps = get_electric_field_mps(
            psi_mps, full_poisson_mpo;
            dv=dv, tolerance=tolerance, maxrank=maxrank, alg="naive",
        )

        accel_mpo_half = get_acceleration_mpo(
            dt/2,              # half step
            Lv,
            M,
            x_grid,
            kv_grid,
            TCI.TensorTrain(electric_field_mps);
            tolerance=tolerance,
            lsb_first=true,
        )
        accel_mpo_half_it = cu(MPO(accel_mpo_half; sites=sites_mpo))

        # apply half acceleration in v-Fourier
        psi_mps = apply(v_fourier_mpo_it, psi_mps; alg="naive", truncate=true, maxdim=maxrank, cutoff=tolerance)
        psi_mps = apply(accel_mpo_half_it, psi_mps; alg="naive", truncate=true, maxdim=maxrank, cutoff=tolerance)
        psi_mps = apply(v_inv_fourier_mpo_it, psi_mps; alg="naive", truncate=true, maxdim=maxrank, cutoff=tolerance)

        # 2) --- full free streaming (dt) ---
        psi_mps = apply(free_stream_mpo_it, psi_mps; alg="naive", truncate=true, maxdim=maxrank, cutoff=tolerance)

        # 3) --- second half acceleration (dt/2) ---
        electric_field_mps = get_electric_field_mps(
            psi_mps, full_poisson_mpo;
            dv=dv, tolerance=tolerance, maxrank=maxrank, alg="naive",
        )

        accel_mpo_half = get_acceleration_mpo(
            dt/2,
            Lv,
            M,
            x_grid,
            kv_grid,
            TCI.TensorTrain(electric_field_mps);
            tolerance=tolerance,
            lsb_first=true,
        )
        accel_mpo_half_it = cu(MPO(accel_mpo_half; sites=sites_mpo))

        psi_mps = apply(v_fourier_mpo_it, psi_mps; alg="naive", truncate=true, maxdim=maxrank, cutoff=tolerance)
        psi_mps = apply(accel_mpo_half_it, psi_mps; alg="naive", truncate=true, maxdim=maxrank, cutoff=tolerance)
        psi_mps = apply(v_inv_fourier_mpo_it, psi_mps; alg="naive", truncate=true, maxdim=maxrank, cutoff=tolerance)

        if step % 5 == 0
            
            tt = TCI.TensorTrain(ITensors.cpu(psi_mps))
            println("Plotting step $step, TT ranks: ", rank(tt))
            f_vals = [tt(origcoord_to_quantics(x_v_grid, (x, v))) for v in v_vals, x in x_vals]
            println("Max f value at step $step: ", maximum(abs.(f_vals)))
            Plots.savefig(
                Plots.heatmap(
                    x_vals,
                    v_vals,
                    abs.(f_vals);
                    xlabel = "x",
                    ylabel = "v",
                    title = "t = $(round(step * dt, digits=3))",
                ),
                "figures/two_stream/phase_space_step$(step).png",
            )
        end
    end
end
