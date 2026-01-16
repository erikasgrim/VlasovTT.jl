import TensorCrossInterpolation as TCI
import Quantics: fouriertransform

function fourier_core_sites(R::Int, target_dim::Int, total_dim::Int, unfoldingscheme::Symbol)
    if total_dim == 1
        return collect(1:R)
    end
    if unfoldingscheme == :interleaved
        return [target_dim + n * total_dim for n in 0:(R - 1)]
    elseif unfoldingscheme == :separate
        start = (target_dim - 1) * R + 1
        return collect(start:(start + R - 1))
    end
    error("Unsupported unfoldingscheme: $(unfoldingscheme)")
end

function stretched_fourier_mpo(
    R::Int,
    target_dim::Int,
    total_dim::Int=2;
    sign::Real=-1.0,
    tolerance::Real=1e-12,
    lsb_first::Bool=false,
    unfoldingscheme::Symbol=:interleaved,
)
    fourier_mpo_1d = quanticsfouriermpo(R; sign=sign, tolerance=tolerance)
    # Flip the MPO ordering if the TT sites are laid out LSB -> MSB.
    fourier_mpo_1d = lsb_first ? TCI.reverse(fourier_mpo_1d) : fourier_mpo_1d
    R_total = R * total_dim

    entry_type = eltype(fourier_mpo_1d[1])
    mpo_cores = Vector{Array{entry_type,4}}(undef, R_total)

    core_sites = fourier_core_sites(R, target_dim, total_dim, unfoldingscheme)
    core_start = total_dim == 1 ? 1 : (target_dim - 1) * R + 1

    for site in 1:R_total
        if site in core_sites
            mpo_site_1d = if total_dim == 1
                site
            elseif unfoldingscheme == :interleaved
                div(site - target_dim, total_dim) + 1
            elseif unfoldingscheme == :separate
                site - core_start + 1
            else
                error("Unsupported unfoldingscheme: $(unfoldingscheme)")
            end
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

function stretched_mpo(
    mpo::TCI.TensorTrain,
    target_dim::Int,
    total_dim::Int=2;
    unfoldingscheme::Symbol=:interleaved,
)
    R = TCI.length(mpo)
    R_total = R * total_dim

    entry_type = eltype(mpo[1])
    mpo_cores = Vector{Array{entry_type,4}}(undef, R_total)

    core_sites = fourier_core_sites(R, target_dim, total_dim, unfoldingscheme)
    core_start = total_dim == 1 ? 1 : (target_dim - 1) * R + 1

    for site in 1:R_total
        if site in core_sites
            mpo_site_1d = if total_dim == 1
                site
            elseif unfoldingscheme == :interleaved
                div(site - target_dim, total_dim) + 1
            elseif unfoldingscheme == :separate
                site - core_start + 1
            else
                error("Unsupported unfoldingscheme: $(unfoldingscheme)")
            end
            mpo_core = mpo[mpo_site_1d]

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

function quanticsfouriermpo_multidim(
    R::Int,
    total_dim::Int=2;
    sign::Real=-1.0,
    algorithm=:TCI,
    tolerance::Real=1e-12,
    lsb_first::Bool=false,
    unfoldingscheme::Symbol=:interleaved,
)
    mpo_list = Vector{TCI.TensorTrain}(undef, total_dim)
    for dim in 1:total_dim
        mpo_list[dim] = stretched_fourier_mpo(
            R,
            dim,
            total_dim;
            sign=sign,
            lsb_first=lsb_first,
            unfoldingscheme=unfoldingscheme,
        )
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
