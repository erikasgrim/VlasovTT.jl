import TensorCrossInterpolation as TCI
import Quantics: fouriertransform

function stretched_fourier_mpo(R::Int, target_dim::Int, total_dim::Int=2; sign::Real=-1.0, tolerance::Real=1e-12, lsb_first::Bool=false)
    fourier_mpo_1d = quanticsfouriermpo(R; sign=sign, tolerance=tolerance)
    # Flip the MPO ordering if the TT sites are laid out LSB -> MSB.
    fourier_mpo_1d = lsb_first ? TCI.reverse(fourier_mpo_1d) : fourier_mpo_1d
    R_total = R * total_dim

    entry_type = eltype(fourier_mpo_1d[1])
    mpo_cores = Vector{Array{entry_type,4}}(undef, R_total)

    fourier_core_sites = [target_dim + n * total_dim for n in 0:(R-1)]

    for site in 1:R_total
        if site in fourier_core_sites
            mpo_site_1d = div(site - target_dim, total_dim) + 1
            mpo_core = fourier_mpo_1d[mpo_site_1d]

        else
            if site == 1
                left_dim = 1
                right_dim = 1

                mpo_core = zeros(entry_type, left_dim, 2, 2, right_dim)
                mpo_core[1, 1, 1, 1] = 1.0
                mpo_core[1, 2, 2, 1] = 1.0

            elseif site == R_total
                left_dim = size(mpo_cores[site-1], 4)
                right_dim = 1
                mpo_core = zeros(entry_type, left_dim, 2, 2, right_dim)
                for chi in 1:left_dim
                    mpo_core[chi, 1, 1, 1] = 1.0
                    mpo_core[chi, 2, 2, 1] = 1.0
                end

            else
                bond_dim = size(mpo_cores[site-1], 4)
                mpo_core = zeros(entry_type, bond_dim, 2, 2, bond_dim)
                for chi in 1:bond_dim
                    mpo_core[chi, 1, 1, chi] = 1.0
                    mpo_core[chi, 2, 2, chi] = 1.0
                end
            end
        end
        mpo_cores[site] = mpo_core
    end

    return TCI.TensorTrain(mpo_cores)
end

function quanticsfouriermpo_multidim(R::Int, total_dim::Int=2; sign::Real=-1.0, algorithm=:TCI, tolerance::Real=1e-12, lsb_first::Bool=false)
    mpo_list = Vector{TensorTrain}(undef, total_dim)
    for dim in 1:total_dim
        mpo_list[dim] = stretched_fourier_mpo(R, dim, total_dim; sign=sign, lsb_first=lsb_first)
    end

    # Contract all the Fourier MPOs together
    fourier_mpo = mpo_list[1]
    for dim in 2:total_dim
        fourier_mpo = TCI.contract(
            mpo_list[dim],
            fourier_mpo;
            algorithm = algorithm,
            tolerance = tolerance,
        )
    end
    return fourier_mpo
end