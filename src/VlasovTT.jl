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
include("write.jl")

export
    k_to_n,
    n_to_k,
    tt_to_mpo,
    mps_to_mpo,
    build_kv_tt,
    interleave_bits,
    frequency_filter,
    PhaseSpaceGrids,
    stretched_fourier_mpo,
    stretched_mpo,
    stretched_mpo_it,
    quanticsfouriermpo_multidim,
    free_streaming_pivots,
    get_free_streaming_mpo,
    acceleration_pivots,
    get_acceleration_mpo,
    get_poisson_mpo,
    get_charge_density,
    get_electric_field_mps,
    get_electric_field_mps_kv,
    AccelerationTCICache,
    gaussian_ic,
    equilibrium_test_ic,
    linear_landau_damping_ic,
    two_stream_instability_ic,
    build_initial_tt,
    SimulationParams,
    strang_step_filtered_TCI!,
    strang_step_unfiltered_TCI!,
    strang_step_filtered_RK4!,
    strang_step_unfiltered_RK4!,
    SolverMPOs,
    build_solver_mpos,
    prepare_itensor_mpos,
    electric_field_energy,
    total_charge,
    kinetic_energy,
    total_momentum,
    ObservablesCache,
    build_observables_cache,
    electric_field_mode_energy,
    PoissonCache,
    build_poisson_cache,
    write_parameters,
    write_data,
    read_data
end
