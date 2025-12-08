module VlasovTT

import TensorCrossInterpolation as TCI
using QuanticsGrids
using QuanticsTCI
using Quantics
using ITensorMPS
using ITensors
using CUDA

include("utilities.jl")
include("grids.jl")
include("fourier.jl")
include("operators/free_streaming.jl")
include("operators/acceleration.jl")
include("operators/poisson.jl")
include("initial_conditions.jl")
include("steppers/strang.jl")
include("solver.jl")
include("observables.jl")

export
    k_to_n,
    n_to_k,
    tt_to_mpo,
    build_kv_tt,
    interleave_bits,
    Theta,
    PhaseSpaceGrids,
    stretched_fourier_mpo,
    stretched_mpo,
    quanticsfouriermpo_multidim,
    free_streaming_pivots,
    get_free_streaming_mpo,
    acceleration_pivots,
    get_acceleration_mpo,
    get_poisson_mpo,
    get_charge_density,
    get_electric_field_mps,
    AccelerationTCICache,
    gaussian_ic,
    equilibrium_test_ic,
    linear_landau_damping_ic,
    two_stream_instability_ic,
    build_initial_tt,
    SimulationParams,
    strang_step_v3!,
    strang_step_v2!,
    strang_step!,
    SolverMPOs,
    build_solver_mpos,
    prepare_itensor_mpos,
    electric_field_energy,
    total_charge,
    kinetic_energy,
    ObservablesCache,
    build_observables_cache,
    PoissonCache,
    build_poisson_cache

end
