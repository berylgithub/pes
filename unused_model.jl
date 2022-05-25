"""
=================================
>>> New bonding features for HxOy: coordinate function and bump functions <<<
=================================
"""
# z(t) coordinate function:
"""
vanilla version is more preferred, use map(f, .) for vector/array, much faster than the vectorized version
"""
function z_coord(t)
    t = max(-1.,min(t,1.))
    t2 = t^2
    z = 0.5 - t*(0.9375 + t2*(0.1875*t2 - 0.625))
    return z
end

"""
vectorized version, for benchmark
"""
function z_coord_vec(t)
    z = zeros(Float64, size(t)) # default values, t ≥ 1
    t .= max.(-1.,(min.(1.,t)))
    t2 = t.^2
    z .= @. 0.5 - t*(0.9375 + t2*(0.1875*t2 - 0.625))
    return z
end

# manual by hand derivatives:
∂z_coord(t) = @. -0.9375 - 0.9375*(t^4) + 1.875*(t^2)
∂∂z_coord(t) = @. -3.75*(t^3) + 3.75*t

# autodiff derivatives:
∂z_coord_auto(t) = map(x -> x[1], map(x -> Zygote.gradient(z_coord, x), t)) 
∂∂z_coord_auto(t) = map(x -> Zygote.hessian(z_coord, x), t)

# 𝑓𝑘(𝑡)  bump functions:
function f_bump(t, k)
    """
    bump function,
    returns: 
        - b, f(t,k), real scalar
    params:
        - t, scalar ∈ R
        - k, scalar ∈ Z
    """
    """
    OLD VER, uncompatible with AD due to explicit constants:
    b = 0. # zeros outside [t_low, t_high]
    if t ≥ k-1. && t ≤ k + 1.
        b = (1. - (t-k)^2)^3
        # take only the positive part:
        b < 0. ? b =-b : b
    end
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
f_k(t_low, t_hi) = collect(ceil(t_low):floor(t_hi)) # generate a list of k ∈ Z ∩ t_low < k < t_hi

sqr(x) = x^2 # abstraction fun of x^2

# main caller for z and bump:
function f_z_bump(R, r_xy, N)
    """
    computes the z(t) and bump function and stacks them into one array for one unique xy atomic pair.
    - returns (n_data, n_d, n_k = k+1) array, where [:,:,k+1] contains z, the rest are bumps.
    d here is actually ⊂ d, since there would be another z(t) and bump calculation if the molecule is heterogeneous, e.g.,
    H2O: (HH, HO, HO) -> (1 c_HH, 2 c_HO), it means the array's sizes would be c_HH: (n_data, 1, k+1), c_HO: (n_data, 2, k+1),
    stacked into z_bumo_mat: (n_data, 3, k+1) total size.
    params:
        - R, subset of matrix of distances, (n_data, n_d) shape, ∈ Float64
        - r_xy, equilibrium distance of pairpot xy, const scalar ∈ real
        - N, hyperparam: num of bumps, scalar ∈ int
    Notes: 
        - should t_low < -1?
        - should z be treated differently from f_k?
    """
    #these need to be moved outside of the function, since for hetero mol these would be needed for other primitive features.
    r2_xy = r_xy^2; r2_hi = maximum(R)^2; r2_low = minimum(R)^2; 
    # R2 = R.^2
    R2 = map(sqr, R)
    # t_xy:
    n_data = size(R2)[1]; n_d = size(R2)[2];
    c_xy = f_c(N, r2_hi, r2_low)
    t_xy = map(i -> f_t(c_xy, r2_xy, (@view R2[:,i])), 1:n_d) # t of equilibrium distance vector{n_d}(vector{n_data})
    t_xy = mapreduce(permutedims, vcat, t_xy)
    t_xy = transpose(t_xy)
    t_low = f_t_scalar(c_xy, r2_xy, r2_low)
    t_hi = f_t_scalar(c_xy, r2_xy, r2_hi)
    k = f_k(t_low, t_hi); k_idx = 1:length(k)
    n_k = length(k); 
    output_mat = Array{Float64}(undef, n_data, n_d, n_k+1) # contains (n_data, d, k+1) array, [:,:,k+1] is z
    # bump:
    range_n_data = 1:n_data; range_n_d = 1:n_d
    for i = k_idx
        out = map(x -> f_bump(x, k[i]), t_xy) # map on matrix
        output_mat[:,:,i] = out
    end

    # z:
    z = map(z_coord, t_xy)
    output_mat[:, :, end] = z;
    #return output_mat, t_xy
    return output_mat
end

function f_z_bump!(output_mat, R, r_xy, N)
    """
    !! UNALLOC VER !!
    computes the z(t) and bump function and stacks them into one array for one unique xy atomic pair.
    - returns (n_data, n_d, n_k = k+1) array, where [:,:,k+1] contains z, the rest are bumps.
    d here is actually ⊂ d, since there would be another z(t) and bump calculation if the molecule is heterogeneous, e.g.,
    H2O: (HH, HO, HO) -> (1 c_HH, 2 c_HO), it means the array's sizes would be c_HH: (n_data, 1, k+1), c_HO: (n_data, 2, k+1),
    stacked into z_bumo_mat: (n_data, 3, k+1) total size.
    params:
        - R, subset of matrix of distances, (n_data, n_d) shape, ∈ Real
        - r_xy, equilibrium distance of pairpot xy, const scalar ∈ Real
        - N, hyperparam: num of bumps, scalar ∈ int
    """
    #these need to be moved outside of the function, since for hetero mol these would be needed for other primitive features.
    r2_xy = r_xy^2; r2_hi = maximum(R)^2; r2_low = minimum(R)^2; 
    # R2 = R.^2
    R2 = map(sqr, R)
    # t_xy:
    n_data = size(R2)[1]; n_d = size(R2)[2];
    c_xy = f_c(N, r2_hi, r2_low)
    t_xy = map(i -> f_t(c_xy, r2_xy, (@view R2[:,i])), 1:n_d) # t of equilibrium distance vector{n_d}(vector{n_data})
    t_xy = mapreduce(permutedims, vcat, t_xy)
    t_xy = transpose(t_xy)
    t_low = f_t_scalar(c_xy, r2_xy, r2_low)
    t_hi = f_t_scalar(c_xy, r2_xy, r2_hi)
    k = f_k(t_low, t_hi); k_idx = 1:length(k)
    
    n_k = length(k); 
    # bump:
    range_n_data = 1:n_data; range_n_d = 1:n_d
    for i = k_idx
        output_mat[:,:,i] = map(x -> f_bump(x, k[i]), t_xy) # map on matrix
    end

    # z:
    output_mat[:, :, end] = map(z_coord, t_xy);
end
