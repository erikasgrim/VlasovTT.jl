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
include("mpo.jl")
include("observables.jl")
include("write.jl")

export
    PhaseSpaceGrids,
    get_electric_field_mps,
    get_electric_field_mps_kv,
    linear_landau_damping_ic,
    two_stream_instability_ic,
    build_initial_tt,
    strang_step_filtered_TCI!,
    strang_step_unfiltered_TCI!,
    strang_step_filtered_RK4!,
    strang_step_unfiltered_RK4!,
    build_solver_mpos,
    prepare_itensor_mpos,
    electric_field_energy,
    total_charge_kv,
    kinetic_energy,
    total_momentum,
    build_observables_cache,
    electric_field_mode_energy,
    write_parameters,
    write_runtimes,
    write_data
end
