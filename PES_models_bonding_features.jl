using Optim, LsqFit # opt libs
using LinearAlgebra # self explanatory
using Zygote, ReverseDiff # autodiff
using StatsBase, DataStructures, DelimitedFiles, MLUtils, BenchmarkTools # utils

"""
New bonding features for HxOy: coordinate function and bump functions
"""

# z(t) coordinate function:
function z_coord(t)
    """
    vanilla version is more preferred, use map(f, .) for vector/array, much faster than the vectorized version
    """
    t = max(-1.,min(t,1.))
    t2 = t^2
    z = 0.5 - t*(0.9375 + t2*(0.1875*t2 - 0.625))
    return z
end

# autodiff derivatives of z(t):
∂z_coord_auto(t) = map(x -> x[1], map(x -> Zygote.gradient(z_coord, x), t)) 
∂∂z_coord_auto(t) = map(x -> Zygote.hessian(z_coord, x), t)

function f_bump(t, k)
    """
    bump function,
    returns: 
        - b, f(t,k), real scalar
    params:
        - t, scalar ∈ R
        - k, scalar ∈ Z
    """
    tlow = k - 1.; thi = k + 1.;
    t = max(tlow, min(thi, t))
    b = (1. - (t-k)^2)^3
    b = max(-b, b)
    return b
end

f_t(c_xy, r2_stat, R2) = c_xy*(R2 .- r2_stat) # takes r2 := r^2 vector, rest are scalars
f_t_scalar(c_xy, r2_stat, r2) = c_xy*(r2 - r2_stat) # scalar version of f_t
f_c(N, r2_hi, r2_low) = N/(r2_hi - r2_low) # scalar ops
f_k(t_low, t_hi) = collect(round(t_low):round(t_hi))[2:end-1] # generate a list of k ∈ Z ∩ t_low < k < t_hi

function f_z_bump(R, r_xy, N)
    """
    computes the z(t) and bump function and concats them into one array for one unique xy atomic pair.
    returns (n_data, d, k+1) array, where [:,:,k+1] contains z, the rest are bumps.
    params:
        - R, subset of matrix of distances, (n_data, n_d) shape, ∈ Float64
        - r_xy, equilibrium distance of pairpot xy, const scalar ∈ real
        - N, hyperparam: num of bumps, scalar ∈ int
    """
    r2_xy = r_xy^2; r2_hi = maximum(R)^2; r2_low = minimum(R)^2;
    R2 = R.^2
    n_data = size(R2)[1]; n_d = size(R2)[2];
    c_xy = f_c(N, r2_hi, r2_low)
    t_xy = map(i -> f_t(c_xy, r2_xy, (@view R2[:,i])), 1:n_d) # t of equilibrium distance vector{n_d}(vector{n_data})
    t_low = f_t_scalar(c_xy, r2_xy, r2_low)
    t_hi = f_t_scalar(c_xy, r2_xy, r2_hi)
    k = f_k(t_low, t_hi); k_idx = convert(Array{Int}, k)

    n_k = length(k); 
    output_mat = Array{Float64}(undef, n_data, n_d, n_k+1) # contains (n_data, d, k+1) array, [:,:,k+1] is z
    # bump:
    range_n_data = 1:n_data; range_n_d = 1:n_d
    for i = k_idx
        for j = range_n_d
            for l = range_n_data 
                output_mat[l,j,i] = f_bump(t_xy[j][l], k[i])
            end
        end
    end

    # z:
    z = map(j -> map(i -> z_coord(t_xy[i][j]), 1:n_d), 1:n_data) # results in (n_data, n_d) array, with n_data in inner loop, hence faster
    z = mapreduce(permutedims, vcat, z)
    output_mat[:, :, end] = z;
    return output_mat
end


@benchmark map(z_coord, [-1., 1.])