using Optim, LsqFit, NLsolve # opt libs
using LinearAlgebra # self explanatory
using Zygote, ReverseDiff, ForwardDiff # autodiff
using Plots, LaTeXStrings # plots
using StatsBase, DataStructures, MLUtils, StaticArrays # data processing utils
using Distributions # sampling distributions
using DelimitedFiles, JSON3, NPZ # file utils
using BenchmarkTools # selfexplanatory
"""
possible efficiency opt: https://github.com/JuliaArrays/StaticArrays.jl # doesnt work for large arrays!!
for checking allocs: 
 - https://github.com/KristofferC/TimerOutputs.jl
 - https://discourse.julialang.org/t/way-to-show-where-memory-allocations-occur/2161
probably helpful for particularly array assignment:
https://discourse.julialang.org/t/help-understanding-performance-with-assignment-to-allocated-matrix/8149

autodiff typeerror: https://discourse.julialang.org/t/forwarddiff-no-method-matching-error/53311

- !! NEED TO CHANGE EVERYTHING TO REAL TYPE FOR forward AD TO WORK !! <<< slow
- need to preallocate everything (like in C++ !!) so no Real allocation instead, Real always allocates due to abstract!!

reversediff Float64 error: https://discourse.julialang.org/t/argumenterror-converting-an-instance-of-reversediff-trackedreal-float64-float32-nothing-to-float32-is-not-defined/69380
"""

"""
========
>>> old model <<<
========
"""
# functional form:
function f_ratpot_2(Θ, R, M)
    #=
    ansatz 1 for diatomic potential
    params:
        - Θ := training parameters, vector ()
        - R := distances, vector
    =#
    # unroll coefficients
    a = Θ[1:M]
    b = Θ[M+1:2*M]
    c = Θ[2*M+1:3*M+4]
    d = Θ[3*M+5:4*M+7]
    
    # b_i ≥ 0 for i > 1:
    t = b[2:M]
    bool = t .< 0.
    t[bool] = -t[bool]
    b[2:M] = t
    
    # d_i ≥ 0 for i > 0:
    bool = d .< 0.
    d[bool] = -d[bool]    
    
    # evaluate P:
    P = c[end-1]
    P = P .* ((R .- a[1]).^2 .+ (b[1] .* R))
    for i=2:M
        P .*= (R .- a[i]).^2 .+ (b[i]*R)
    end

    
    # eval Q:
    Q = (R .+ d[end]).*R
    for i=1:M+2
        Q .*= (R .- c[i]).^2 .+ d[i].*R
    end
    
    # eval potential:
    V = c[end] .+ (P ./ Q)
    return V
end


"""
=========================
>>> computation utils <<<
=========================
"""
tsqrt(x) = x > 0. ? √x : √1e-6 # truncated sqrt, sqrt(x) if x > 0 else sqrt(1e-6)
tlog(x) = log(max(0.1, x)) # truncated log 

"""
use distance to coord implementation in Py for now. Load the files using NPZ
"""
function atom_indexer(n_atom)
    """
    generates array of coordination indexes for Y[i] vector, which depends on num_atom, e.g.:
    num_atom = 3:  [[1,2],[1,3],[2,3]]
                     b_1j  b_2j b_3j
    num_atom = 4: [[1,2,3],[1,4,5],[2,4,6],[3,5,6]]
                    b_1j     b_2j   b_3j     b_4j
    in form of Matrix{Int}(n_atom-1,n_atom), ordered by column (column dominant)
    """
    init_idx = collect(n_atom:-1:2)
    group_idx = []
    start = 1
    for idx=init_idx
        list_index = collect(start:idx+start-2)
        push!(group_idx, list_index)
        start += idx - 1
    end
    coord_idx = []
    for i=range(1, step=1, n_atom)
        coord = []
        if i == 1
            coord = group_idx[1]
            push!(coord_idx, coord)
        elseif i == n_atom
            enumerator = collect(range(i-1, step=-1, 1))
            counter = 1
            for num=enumerator
                push!(coord, group_idx[counter][num])
                counter += 1
            end
            push!(coord_idx, coord)
        else
            enumerator = collect(range(i-1, step=-1, 1))
            counter = 1
            for num=enumerator
                push!(coord, group_idx[counter][num])
                counter += 1
            end
            append!(coord, group_idx[counter])
            push!(coord_idx, coord)
        end
    end
    coord_idx = transpose(mapreduce(permutedims, vcat, coord_idx))
    coord_idx = convert(Matrix{Int}, coord_idx)
    return coord_idx
end



"""
==================
>>> objectives <<<
==================
"""

f_RMSE(X, Y) = √(sum((X .- Y) .^ 2)/length(X))

function f_least_squares(f_eval, Y, f_args...)
    Y_pred = f_eval(f_args...)
    res = sum((Y .- Y_pred).^2)
    return res
end

function f_least_squares_vec(f_eval, Y, f_args...)
    """
    vector version residual for nonlinear leastsquares solvers
    """
    Y_pred = f_eval(f_args...)
    res = (Y .- Y_pred).^2
    return res
end

"""
=================================
>>> New bonding features for HxOy: coordinate function and bump functions <<<
=================================
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

function z_coord_vec(t)
    """
    vectorized version, for benchmark
    """
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

"""
===================
>>> Tchebyshev pol primitive features <<<
===================
"""
function t_R_fun(R, R_up, R_low, e)
    """
    sub bond strength, similar(R) when using map()
    """
    R2 = R^2
    return ((R2 - R_low^2)/(R_up^2 - R2))^e
end


function s_bond_strength(R, R_up, R_low, t, t0)
    """
    s_{ij} = s(R_{ij}), similar(R)
    t0 = t_R_fun(Rm, R_up, R_low, e)
    if R_m = R and R_low < R_m < R_up, then bond_strength_s = 0.5
    trainable parameters: (R_low, R_m, R_up)
    """
    s = 0.
    if R < R_low
        s = 1.
    elseif R_low ≤ R ≤ R_up
        s = t0/(t+t0)
    end
    return s
end

s_dash_f(s) = 2 - 4 * s
function p_tchebyshev_pol(out, deg, s, s_dash)
    """
    bonding pol in tchebyshev term
    returns scalar, need to fill deg == 1 manually since it's unreachable
    params:
        - out, result storage, containing vector length = deg ∈ Real
        - deg, maximum pol degree, scalar ∈ Int
        - s, scalar ∈ Real
        - s_dash, scalar ∈ Real
    """
    if deg == 1
        out[deg] = s
        return s
    elseif deg == 2
        out[deg] = s*(1-s)
        return out[deg]
        #return s*(1-s)
    elseif deg == 3 # p_3 = s'p_2
        out[deg] = s_dash*p_tchebyshev_pol(out, 2, s, s_dash)
        return out[deg]
        #return s_dash*p_tchebyshev_pol(2, s, s_dash)
    elseif deg > 3 # p_{d+1} = s'p_d - p_{d-1}
        out[deg] = s_dash*p_tchebyshev_pol(out, deg-1, s, s_dash) - p_tchebyshev_pol(out, deg-2, s, s_dash)
        return out[deg]
        #return s_dash*p_tchebyshev_pol(deg-1, s, s_dash) - p_tchebyshev_pol(deg-2, s, s_dash)
    end
end

function f_b_feature(R, R_up, R_m, R_low, max_deg, e)
    """
    computes the primitive feature in terms of bonding polynomial (tchebyshev pol)
    returns array, shape = (n_data, n_d, n_k), where n_k = max_deg
    params:
        - R, matrix of distances, shape = (n_data, n_d) ∈ Real
        - R_up, R_m, R_low, tuning params, scalars ∈ Real 
        - max_deg, e, hyperparams, scalars ∈ Real
    """
    n_data, n_d = size(R)
    t0 = t_R_fun(R_m, R_up, R_low, e) # const
    t = map(r -> t_R_fun(r, R_up, R_low, e), R) # matrix similar(R), iter = 1, map() is better
    s = s_bond_strength.(R, R_up, R_low, t, t0) # matrix similar(R), iter > 1, vectorize is better 
    s_dash = map(s_dash_f, s) # matrix similar(R)
    b = Array{Float64}(undef, max_deg, n_data, n_d) # array (n_data, n_d, n_k)
    n_data, n_d = size(R)
    rg_d = 1:n_d; rg_data = 1:n_data
    for j=rg_d
        @simd for i = rg_data
            @inbounds p_tchebyshev_pol((@view b[:,i,j]), max_deg, s[i,j], s_dash[i,j])
        end
    end
    b[1,:,:] = s # fill the index 1 array manually, 
    b = permutedims(b, [2,3,1])
    return b
end

"""
======================
>>> Bonding features:  𝑈,𝑌(𝑧(𝑡),𝑓bump),𝐺(𝑧(𝑡),𝑓bump) <<<
======================
"""
function f_Y_coord(z_bump_mat, idxer)
    """
    Y_d[i] = sum_{j neq i}b_{ijd}
    returns an array with shape = (n_data, n_k, n_atom)
    params:
        - z_bump_mat, array containing z(t) and bump functions, shape = (n_data, n_d, n_k) ∈ Float64
        - idxer, matrix containing the indexer required for 2-body sums, shape = (n_atom-1, n_atom)
    """
    n_data, n_d, n_k = size(z_bump_mat); n_atom = size(idxer)[2]
    Y_mat = Array{Float64}(undef, (n_data, n_k, n_atom))
    #Y_mat = zeros(n_data, n_k, n_atom)
    # 2-body sums for each atom:
    for i=1:n_atom
        atom_arr = @view z_bump_mat[:, idxer[:,i],:]
        Y_mat[:,:,i] = sum(atom_arr, dims=2)
    end
    return Y_mat
end

function f_Δcoord(X)
    """
    X_j - X_i, i \neq j
    returns: Δ array containing the differences of the coordinates, shape = (n_data, n_elem=3, n_d) ∈ Float64
    params:
    - X, matrix containing coordinates of the atoms, shape = (n_data, n_atom, 3)
    """
    n_data, n_atom, n_elem = size(X)
    n_d = Int((n_atom^2 - n_atom)/2)
    Δ = Array{Float64}(undef, (n_data, n_elem, n_d))
    X = permutedims(X, [1,3,2]) # move axis, becomes (n_data, 3, n_atom)
    rg_atom = 1:n_atom
    d = 1
    for i=rg_atom
        @simd for j=rg_atom
            if i<j
                x_j = @view X[:,:,j]; x_i = @view X[:,:,i]
                Δ[:,:,d] = x_j - x_i #X_j - X_i
                d += 1
            end
        end
    end
    #Δ = permutedims(Δ, [2,1,3]) # move axis again, becomes (3, n_data, n_atom)
    return Δ
end

svmul(c, x) = c*x
function f_r_orient_vec(z_bump, Δ, idxer)
    """
    r_d[i] ∈ R^3 = sum(z_bump_ij*Δ_ij) -> scalar*vector
    returns: array of r, shape = (3, n_data, n_k, n_atom)
    params:
        - z_bump_mat, array containing z(t) and bump functions, shape = (n_data, n_d, n_k) ∈ Float64
        - Δ, array containing the differences of the coordinates, shape = (n_data, 3, n_d) ∈ Float64
        - idxer, matrix containing the atomic indexer, shape = (n_atom-1,n_atom) ∈ Int
    """
    n_data, n_d, n_k = size(z_bump); n_atom = size(idxer)[2]
    temp_out = Array{Float64}(undef, 3, n_data, n_d, n_k)
    #temp_out = @SArray zeros(n_data, n_d, n_k)
    rg_k = 1:n_k; rg_j = 1:n_d; rg_i=1:n_data
    # vanilla loop for z[i,j,k]*Δ[i,:,j]:
    Δ = permutedims(Δ, [2,1,3]) # put vector length to front
    for k=rg_k
        for j=rg_j
            @simd for i=rg_i
                @inbounds temp_out[:,i,j,k] .= z_bump[i,j,k]*(@view Δ[:,i,j])
            end
        end
    end
    # ∑z*Δ, 2-body sums for each atom:
    rk = Array{Float64}(undef, 3, n_data, n_k, n_atom)
    temp_out = permutedims(temp_out, [1,2,4,3]) # move the op to the back
    @simd for i=1:n_atom
        atom_arr = @view temp_out[:,:,:,idxer[:,i]]
        @inbounds rk[:,:,:,i] = sum(atom_arr, dims=4)
    end
    return rk
end

function f_G_mat(rk)
    """
    G[i]_k1k2 ∈ R = r_k1[i] ⋅ r_k2[i]
    returns:
        - G, array from dot product of rk, shape = (n_data, n_k, n_k, n_atom)
    params:
        - rk, array of orientation vectors, shape = (3, n_data, n_k, n_atom)
    """
    _, n_data, n_k, n_atom = size(rk)
    G = Array{Float64}(undef, n_data, n_k, n_k, n_atom)
    rg_k = 1:n_k
    for k1=rg_k
        @simd for k2=rg_k
            if k1 <= k2
                rk1 = @view rk[:,:,k1,:]; rk2 = @view rk[:,:,k2,:]
                G[:, k1, k2, :] = sum(rk1.*rk2, dims=1)
            else
                G[:, k1, k2, :] = @view G[:, k2, k1, :]
            end
        end
    end
    return G
end

function V_ref_pairpot(R, C, R_h, R_C, R_0, g)
    """
    trainable pair potential, all inputs and outputs are scalars, use f.() for array of R !! # somehow this allocates less
    reverse AD compatibiliy ??
    """
    V = 0.
    if R ≤ R_h
        V = Inf
    elseif R_h ≤ R ≤ R_C
        R2 = R^2
        V = -C*(R_C^2 - R2)^g * ((R2 - R_0^2)/(R2 - R_h^2))
    end
    return V
end

function f_U_bas(R, idxer, arg_vref...)
    """
    U = ∑V_ij
    returns matrix (n_data, n_atom) ∈ Float64
    params:
        - R, matrix of distances, shape = (n_data, n_d) ∈ Float64
        - idxer, matrix containing the atomic indexer, shape = (n_atom-1,n_atom) ∈ Int
        - all params of V_ref_pairpot (arg_vref...)
    """
    n_data = size(R)[1]; n_atom = size(idxer)[2]
    Vref = V_ref_pairpot.(R, arg_vref...)
    U = Matrix{Float64}(undef, n_data, n_atom)
    #U = zeros(n_data, n_atom)
    @simd for i=1:n_atom
        Vsub = @view Vref[:, idxer[:,i]]
        U[:, i] = sum(Vsub, dims=2)
    end
    U = U./maximum(abs.(U)) # scale U, by U:=U/max(abs(U))
    return U
end

function f_Φ(U, Y, G, n_basis)
    """
    constructs Φ array of basis (mathematically, a matrix), all in and outputs' arrays ∈ Float64
    returns: array with shape = (n_data, n_basis, n_atom)
    params:
        - U, array, shape = (n_data, n_atom)
        - Y, array, shape = (n_data, n_k, n_atom)
        - G, array, shape = (n_data, n_k, n_k, n_atom)
        - n_basis, number of basis, scalar ∈ Int
    """
    n_data, n_atom = size(U)
    Φ = Array{Float64}(undef, n_data, n_basis, n_atom)
    # manually input all basis:
    # degree 1:
    Φ[:,1,:] .= U
    Φ[:,2,:] .= (@view Y[:,1,:])
    # degree 2:
    Φ[:,3,:] .= U .* (@view Y[:,1,:])
    Φ[:,4,:] .= (@view Y[:, 1,:]).^2
    Φ[:,5,:] .= (@view Y[:, 2,:])
    Φ[:,6,:] .= (@view G[:, 1,1,:])
    # degree 3:
    Φ[:, 7,:] .= U .* (@view Φ[:, 4,:]) 
    Φ[:, 8,:] .= U .* (@view Y[:, 2,:]) 
    Φ[:, 9,:] .= U .* (@view G[:, 1, 1,:])
    Φ[:, 10,:] .= (@view Φ[:, 4,:]) .* (@view Y[:, 1,:]) 
    Φ[:, 11,:] .= (@view Y[:, 1,:]) .* (@view Y[:, 2,:]) 
    Φ[:, 12,:] .= (@view Y[:, 3,:]) 
    Φ[:, 13,:] .= (@view G[:, 1, 1,:]) .* (@view Y[:, 1,:]) 
    Φ[:, 14,:] .= (@view G[:, 1, 2,:])
    # degree 4:
    Φ[:, 15,:] .= U .* (@view Φ[:, 10,:])
    Φ[:, 16,:] .= U .* (@view Y[:, 1,:]) .* (@view Y[:, 2,:])
    Φ[:, 17,:] .= U .* (@view Y[:, 3,:])
    Φ[:, 18,:] .= U .* (@view G[:, 1,1,:]) .* (@view Y[:, 1,:])
    Φ[:, 19,:] .= U .* (@view G[:, 1,2,:])
    Φ[:, 20,:] .= (@view Φ[:, 10,:]) .* (@view Y[:, 1,:]); #Y_1^4
    Φ[:, 21,:] .= (@view Φ[:, 4,:]) .* (@view Y[:, 2,:]); #Y_1^2Y_2
    Φ[:, 22,:] .= (@view Y[:, 1,:]) .* (@view Y[:, 3,:]);
    Φ[:, 23,:] .= (@view Y[:, 2,:]) .^ 2; #Y_2^2
    Φ[:, 24,:] .= (@view Y[:, 4,:]);
    Φ[:, 25,:] .= (@view G[:, 1,1,:]) .* (@view Φ[:, 4,:]);
    Φ[:, 26,:] .= (@view G[:, 1,1,:]) .* (@view Y[:, 2,:]);
    Φ[:, 27,:] .= (@view G[:, 1,1,:]) .^ 2;
    Φ[:, 28,:] .= (@view G[:, 1,2,:]) .* (@view Y[:, 1,:]);
    Φ[:, 29,:] .= (@view G[:, 1,3,:]);
    Φ[:, 30,:] .= (@view G[:, 2,2,:]);
    # degree 5:
    Φ[:, 31, :] .= U .* (@view Φ[:, 20, :]);
    Φ[:, 32, :] .= U .* (@view Φ[:, 4, :]) .* (@view Y[:, 2, :]);
    Φ[:, 33, :] .= U .* (@view Φ[:, 4, :]) .* (@view G[:, 1,1, :]);
    Φ[:, 34, :] .= U .* (@view Y[:, 1, :]) .* (@view Y[:, 3, :]);
    Φ[:, 35, :] .= U .* (@view Φ[:, 28, :]);
    Φ[:, 36, :] .= U .* (@view Φ[:, 23, :]);
    Φ[:, 37, :] .= U .* (@view Φ[:, 26, :]);
    Φ[:, 38, :] .= U .* (@view Y[:, 4, :]);
    Φ[:, 39, :] .= U .* (@view Φ[:, 27, :]);
    Φ[:, 40, :] .= U .* (@view G[:, 1,3, :]);
    Φ[:, 41, :] .= U .* (@view G[:, 2,2, :]);
    Φ[:, 42, :] .= (@view Φ[:, 20, :]) .* (@view Y[:, 1, :]); # Y_1^5
    Φ[:, 43, :] .= (@view Φ[:, 10, :]) .* (@view Φ[:, 4, :]);
    Φ[:, 44, :] .= (@view Φ[:, 10, :]) .* (@view G[:, 1,1, :]);
    Φ[:, 45, :] .= (@view Φ[:, 4, :]) .* (@view Y[:, 3, :]);
    Φ[:, 46, :] .= (@view Φ[:, 4, :]) .* (@view G[:, 1,2, :]);
    Φ[:, 47, :] .= (@view Y[:, 1, :]) .* (@view Φ[:, 23, :]);
    Φ[:, 48, :] .= (@view Φ[:, 11, :]) .* (@view G[:, 1,1, :]);
    Φ[:, 49, :] .= (@view Y[:, 1, :]) .* (@view Y[:, 4, :]);
    Φ[:, 50, :] .= (@view Φ[:, 13, :]) .* (@view G[:, 1,1, :]);
    Φ[:, 51, :] .= (@view Y[:, 1, :]) .* (@view G[:, 1,3, :]);
    Φ[:, 52, :] .= (@view Y[:, 1, :]) .* (@view G[:, 2,2, :])
    Φ[:, 53, :] .= (@view Y[:, 2, :]) .* (@view Y[:, 3, :])
    Φ[:, 54, :] .= (@view Y[:, 2, :]) .* (@view G[:, 1,2, :])
    Φ[:, 55, :] .= (@view Y[:, 3, :]) .* (@view G[:, 1,1, :])
    Φ[:, 56, :] .= (@view Y[:, 5, :])
    Φ[:, 57, :] .= (@view G[:, 1,1, :]) .* (@view G[:, 1,2, :])
    Φ[:, 58, :] .= (@view G[:, 1,4, :])
    Φ[:, 59, :] .= (@view G[:, 2,3, :])
    return Φ
end

"""
===============
>>> Quadratic models <<<
===============
"""
function f_A(θ, ϕ)
    """
    rational quadratic model of A form
    returns a vector, shape = n_data ∈ Float64
    params:
        - θ ⊂ Θ, subset of the parameter matrix, shape = (n_basis, 2) ∈ Float64
        - ϕ:=Φ[i] ⊂ Φ, subset of the basis array (indexed by atom), shape = (n_data, n_basis) ∈ Float64
    """
    n_data, n_basis = size(ϕ)
    # using matrix*vector mult:
    numer = ϕ * (@view θ[:,1])
    denom = ϕ * (@view θ[:,2])
    denom = denom.^2 .+ 1.
    return numer ./ denom
end

function f_T0(θ, ϕ)
    """
    rational quadratic model of T0 form, numer^2
    returns a vector, shape = n_data ∈ Float64
    params:
        - θ ⊂ Θ, subset of the parameter matrix, shape = (n_basis, 2) ∈ Float64
        - ϕ:=Φ[i] ⊂ Φ, subset of the basis array (indexed by atom), shape = (n_data, n_basis) ∈ Float64
    """
    n_data, n_basis = size(ϕ)
    numer = ϕ * (@view θ[:,1])
    denom = ϕ * (@view θ[:,2])
    denom = denom.^2 .+ 1.
    return (numer).^2 ./ denom
end

function f_energy(Θ, Φ)
    """
    the sum of atomic energy terms. ϵ = ∑ϵ0[i], where ϵ0[i] := A[i] - √(B[i] + C[i])
    returns a vector, shape = n_data ∈ Float64
    params:
        - Θ, tuning parameter matrix, shape = (n_basis, 6) ∈ Float64
        - Φ, basis array, shape = (n_data, n_basis, n_atom) ∈ Float64
    """
    n_data, n_basis, n_atom = size(Φ)
    # compute partial energy terms:
    A = Matrix{Float64}(undef, n_data, n_atom)
    B = similar(A); C = similar(A)
    for i=1:n_atom
        A[:, i] .= f_A((@view Θ[:,1:2]), (@view Φ[:,:,i])) # A term
        B[:, i] .= f_T0((@view Θ[:,3:4]), (@view Φ[:,:,i])) # B term
        C[:, i] .= f_T0((@view Θ[:,5:6]), (@view Φ[:,:,i])) # C term
    end
    # compute atomic terms:
    ϵ0 = Matrix{Float64}(undef, n_data, n_atom)
    for i=1:n_atom
        ϵ0[:, i] .= (@view A[:, i]) .- .√((@view B[:,i]) .+ (@view C[:,i]))
    end
    ϵ = sum(ϵ0, dims=2) # sum all atomic terms
    return ϵ
end

"""
===============
>>> Main fun evals for z and bumps <<<
===============
"""
function f_pot_bond(Θ, C, R_h, R_C, R_0, 
        R, X, 
        r_xy, N, n_atom, n_basis, 
        idxer, g=6)
    """
    function evaluation of V(.) using bonding features.
    returns V, vector of potential energy, shape = n_data ∈ Float64
    params:
    - Θ, matrix of tuning parameters for the quadratic models, shape = (n_basis, 6) ∈ Float64
    - C, R_h, R_C, R_0, tuning parameters for U basis, all scalars ∈ Float64
    - R, matrix of distances, shape = (n_data, n_d) ∈ Float64
    - X, array of atomic coordinates, shape = (n_data, n_atom, 3) ∈ Float64
    - r_xy, fixed param, equilibrium distance of XY atomic pair, scalar ∈ Float64
    - N, hyperparam, number of bump functions + 1, scalar ∈ Int
    - n_atom, n_basis, self explanatory, scalar ∈ Int
    - idxer, matrix of atomic indexes, shape = (n_atom-1, n_atom) ∈ Float64
    - g, hyperparam for U, optional arg, scalar ∈ Float64
    """
    # U, pair potential feature:
    U = f_U_bas(R, idxer, C, R_h, R_C, R_0, g)

    # bump and coordination functions:
    zb = f_z_bump(R, r_xy, N)

    # Coordination array:
    Y = f_Y_coord(zb, idxer)

    # Δ matrix sub feature:
    Δ = f_Δcoord(X)
    # orientation vector sub feature:
    rk = f_r_orient_vec(zb, Δ, idxer)
    # Gram matrices feature:
    G = f_G_mat(rk)

    # basis array:
    Φ = f_Φ(U, Y, G, n_basis)

    # compute total energy:
    V = f_energy(Θ, Φ)

    return V 
end

function param_converter(ρ)
    """
    convert C, R_h, R_C, R_0, to unconstrained
    """
    π = Vector{Float64}(undef, length(ρ))
    π[1] = tlog(ρ[1])/20 # log(C)/20
    π[2] = tlog(ρ[2])/20 # log(R_h)/20
    π[4] = tsqrt(ρ[4]) # sqrt(R_0)
    π[3] = tsqrt(ρ[3] - ρ[4]) # sqrt(R_C - R_0)

    return π
end

function param_inverter(ρ)
    """
    revert C, R_h, R_C, R_0, to initial, comply to R_h ≤ R_0 ≤ R_C
    """
    π = Vector{Float64}(undef, length(ρ))
    π[1] = exp(20. * ρ[1]) # C
    π[2] = exp(20. * ρ[2]) # R_h
    π[4] = ρ[4]^2  # R_0
    π[3] = ρ[3]^2 + π[4] # R_C

    return π
end

function f_eval_wrapper(Θ_vec, arg_f...)
    """
    wrapper for function evaluation, so that the tuning param is a long vector
    returns: V, vector of energy, shape = n_data
    params:
    - Θ_vec, tuning parameters, vector, shape = n_basis*6 + 4
    """
    n_basis = arg_f[6]
    Θ = Matrix{Float64}(undef, n_basis, 6)
    for i=1:6
    Θ[:, i] = Θ_vec[((i-1)*n_basis) + 1 : i*n_basis]
    end
    # convert then invert U parameters:
    ρ = param_converter(Θ_vec[[end-3, end-2, end-1, end]])
    C, R_h, R_C, R_0 = param_inverter(ρ)
    V = f_pot_bond(Θ, C, R_h, R_C, R_0, arg_f...)
    return vec(V) # convert to vector
end

"""
===============
>>> Main fun evals for Tche𝑏yshev bonding functions <<<
===============
"""
function f_pot_bond_b(Θ, C, R_h, R_low, R_0, R_m, R_up, R_C, 
        R, X, 
        n_atom, n_basis, max_deg,
        idxer, g=6, e=3)
    """
    function evaluation of V(.) using bonding features.
    returns V, vector of potential energy, shape = n_data ∈ Real
    params:
    - Θ, matrix of tuning parameters for the quadratic models, shape = (n_basis, 6) ∈ Real
    - C, R_h, R_C, R_0, tuning parameters for U basis, all scalars ∈ Real
    - R, matrix of distances, shape = (n_data, n_d) ∈ Real
    - X, array of atomic coordinates, shape = (n_data, n_atom, 3) ∈ Real
    - r_xy, fixed param, equilibrium distance of XY atomic pair, scalar ∈ Real
    - N, hyperparam, number of bump functions + 1, scalar ∈ Int
    - n_atom, n_basis, self explanatory, scalar ∈ Int
    - idxer, matrix of atomic indexes, shape = (n_atom-1, n_atom) ∈ Real
    - g, hyperparam for U, optional arg, scalar ∈ Real
    """
    # U, pair potential feature:
    U = f_U_bas(R, idxer, C, R_h, R_C, R_0, g)

    # tchebyshev primitive feature:
    b = f_b_feature(R, R_up, R_m, R_low, max_deg, e)

    # Coordination array:
    Y = f_Y_coord(b, idxer)

    # Δ matrix sub feature:
    Δ = f_Δcoord(X)
    # orientation vector sub feature:
    rk = f_r_orient_vec(b, Δ, idxer)

    # Gram matrices feature:
    G = f_G_mat(rk)

    # basis array:
    Φ = f_Φ(U, Y, G, n_basis)

    # compute total energy:
    V = f_energy(Θ, Φ)

    return V 
end

function param_converter_b(ρ)
    """
    convert C, R_h, R_low, R_m, R_0, R_up, R_C, to unconstrained
    """
    π = Vector{Float64}(undef, length(ρ))
    π[1] = tlog(ρ[1])/20 # log(C)/20
    π[2] = tlog(ρ[2])/20 # log(R_h)/20
    π[3] = tsqrt(ρ[3]) # sqrt(R_low)
    π[4] = tsqrt(ρ[4] - ρ[3]) # sqrt(R_0 - R_low)
    π[5] = tsqrt(ρ[5] - ρ[4]) # sqrt(R_m - R_0)
    π[6] = tsqrt(ρ[6] - ρ[5]) # sqrt(R_up - R_m)
    π[7] = tsqrt(ρ[7] - ρ[6]) # sqrt(R_C - R_up)
    return π
end

function param_inverter_b(ρ)
    """
    revert C, R_h, R_low, R_m, R_0, R_up, R_C, to initial, comply to R_h ≤ R_low ≤ R_0 ≤ R_m ≤ R_up ≤ R_C
    """
    π = Vector{Float64}(undef, length(ρ))
    π[1] = exp(20. * ρ[1]) # C
    π[2] = exp(20. * ρ[2]) # R_h
    π[3] = ρ[3]^2 # R_low
    π[4] = ρ[4]^2 + π[3] # R_0
    π[5] = ρ[5]^2 + π[4] # R_m
    π[6] = ρ[6]^2 + π[5] # R_up
    π[7] = ρ[7]^2 + π[6] # R_C
    return π
end

function f_eval_wrapper_b(Θ_vec, arg_f...)
    """
    wrapper for function evaluation, so that the tuning param is a long vector
    returns: V, vector of energy, shape = n_data
    params:
    - Θ_vec, tuning parameters, vector, shape = n_basis*6 + 7
    """
    n_basis = arg_f[4]
    Θ = Matrix{Float64}(undef, n_basis, 6)
    for i=1:6
    Θ[:, i] = Θ_vec[((i-1)*n_basis) + 1 : i*n_basis]
    end
    # convert then invert U parameters:
    ρ = param_converter_b(Θ_vec[end-6:end])
    C, R_h, R_low, R_0, R_m, R_up, R_C = param_inverter_b(ρ)
    V = f_pot_bond_b(Θ, C, R_h, R_low, R_0, R_m, R_up, R_C, arg_f...)
    return vec(V) # convert to vector
end


"""
=== MAIN CALLERS
"""
function benchmarktest()
    # redefine data
    H_data = readdlm("data/h3/h3_data.txt")
    # load atomic coordinates:
    H_coord = npzread("data/h3/h3_coord.npy")
    R = H_data[:,1:end-1]; V = H_data[:, end-1]
    n_data = size(R)[1]
    idxes = shuffleobs(1:n_data) # shuffle indexes
    id_train, id_test = splitobs(idxes, at=0.8) # split train and test indexes
    # split data by index:
    X = H_coord
    R_train = R[id_train,:]; V_train = V[id_train];
    R_test = R[id_test,:]; V_test = V[id_test]
    X_train = H_coord[id_train,:,:]; X_test = H_coord[id_test,:,:]
    # init params:
    n_atom, n_basis, g = (3, 59, 6.)
    Θ = rand(Distributions.Uniform(-1.,1.), n_basis, 6)
    C, R_h, R_C, R_0 = (1., 0.01, 2., .9,)
    r_xy, N = (1.4172946, 5)
    idxer = atom_indexer(n_atom)
    V = f_pot_bond(Θ, C, R_h, R_C, R_0, R, X, r_xy, N, n_atom, n_basis, idxer, g)
    display(V)
    Θ_vec = vcat(Θ[:], [C, R_h, R_C, R_0])
    V = f_eval_wrapper(Θ_vec, R, X, r_xy, N, n_atom, n_basis, idxer, g)
    display(V)
    # only callable directly in main:
    @benchmark f_pot_bond(Θ, C, R_h, R_C, R_0, R, X, r_xy, N, n_atom, n_basis, idxer, g)
end

function opttest()
    # redefine data here for convenience
    H_data = readdlm("data/h3/h3_data.txt")
    # load atomic coordinates:
    H_coord = npzread("data/h3/h3_coord.npy")
    R = H_data[:,1:end-1]; V = H_data[:, end-1]
    n_data = size(R)[1]
    idxes = shuffleobs(1:n_data) # shuffle indexes
    id_train, id_test = splitobs(idxes, at=0.8) # split train and test indexes
    # split data by index:
    X = H_coord
    R_train = R[id_train,:]; V_train = V[id_train];
    R_test = R[id_test,:]; V_test = V[id_test]
    X_train = H_coord[id_train,:,:]; X_test = H_coord[id_test,:,:]
    # param set:
    siz = 50
    sub_R = R_train[1:siz,:]
    sub_V = V_train[1:siz]
    sub_X = X_train[1:siz, :, :]

    r_xy, N = (1.4172946, 5)
    n_atom, n_basis, g = (3, 59, 6.)
    idxer = atom_indexer(n_atom)

    Θ_vec = rand(Distributions.Uniform(-1.,1.), n_basis*6 + 4)
    #V = f_eval_wrapper(Θ_vec, sub_R, sub_X, r_xy, N, n_atom, n_basis, idxer, g)
    #display(V)
    #f_least_squares(f_eval_wrapper, sub_V, Θ_vec, sub_R, sub_X, r_xy, N, n_atom, n_basis, idxer, g)
    res = optimize(Θ -> f_least_squares(f_eval_wrapper, sub_V, Θ, sub_R, sub_X, 
                                    r_xy, N, n_atom, n_basis, idxer, g),
                Θ_vec, BFGS(),
                Optim.Options(iterations = 100, show_trace=true);
                )
    # check RMSE:
    V_pred = f_eval_wrapper(res.minimizer, sub_R, sub_X, r_xy, N, n_atom, n_basis, idxer, g)
    println(f_RMSE(sub_V, V_pred))
    for i=1:length(sub_V)
        println(sub_V[i]," ",V_pred[i])
    end
end

function opttest2()
    n_atom, n_basis, g, e = (3, 59, 6, 3)
    r_xy, N, max_deg = (1.4172946, 5, 5)
    idxer = atom_indexer(n_atom)

    H_datat = readdlm("data/h3/h3_data.txt")
    # load atomic coordinates:
    Xt = npzread("data/h3/h3_coord.npy")
    Rt = H_datat[:,1:end-1]; Vt = H_datat[:, end]
    siz = 100
    sub_Rt = Rt[1:siz,:];
    sub_Vt = Vt[1:siz];
    sub_Xt = Xt[1:siz, :, :];
    Θ_vec = rand(Distributions.Uniform(-1.,1.), n_basis*6 + 7)
    #Θ_vec = ones(n_basis*6 + 7)
    #Θ_vec[end-6:end] = [1., 0.01, 0.02, 0.5, .9, 10., 11.]
    """
    Θ_opt = readdlm("params/h3/c_params_060422_full_fold1_5e-4_5e-3_umaxabs.out")
    # >>> parameter correspondence to py ver!! <<<:
    temp = Θ_opt[1:7]
    Θ_opt[1:end-7] = Θ_opt[8:end]
    Θ_opt[end-6:end] = temp
    """
    V = f_eval_wrapper_b(Θ_vec, sub_Rt, sub_Xt, n_atom, n_basis, max_deg, idxer, g, e)
    println(V)
    f_least_squares(f_eval_wrapper_b, sub_Vt, Θ_vec, sub_Rt, sub_Xt, n_atom, n_basis, max_deg, idxer, g, e)

    # test direct optimization!!:
    """
    # bumps:
    res = optimize(Θ -> f_least_squares(f_eval_wrapper, sub_V, Θ, sub_R, sub_X, 
                                        r_xy, N, n_atom, n_basis, idxer, g),
                    Θ_vec, LBFGS(m=10),
                    Optim.Options(iterations = 5, show_trace=true);
                    #autodiff = :forward
                    )
    # tchebyshev:
    res = optimize(Θ -> f_least_squares(f_eval_wrapper_b, sub_Vt, 
                                        Θ_vec, sub_Rt, sub_Xt, n_atom, n_basis, max_deg, idxer, g, e),
                    Θ_vec,
                    Optim.Options(iterations = 100, show_trace=true);
                    autodiff = :forward
                    )
    # NLS solver:
    
    """
    #res = LsqFit.curve_fit((R_train, θ) -> f_ratpot_2(θ, R_train, M), J_f, R_train, V_train, θ, show_trace=false, maxIter=100)
    res = LsqFit.curve_fit((R, θ) -> f_eval_wrapper_b(θ, R, sub_Xt, n_atom, n_basis, max_deg, idxer, g, e),
                            sub_Rt, sub_Vt, Θ_vec, show_trace=true, maxIter=100)
    # save res to file:
    #writedlm("minimizer_H3_50data.csv", res.minimizer)
    #x = readdlm("minimizer_H3_50data.csv", '\t')
    V_pred = f_eval_wrapper(res.param, sub_Rt, sub_Xt, r_xy, N, n_atom, n_basis, idxer, g)
    println(f_RMSE(sub_Vt, V_pred))
    for i=1:length(sub_Vt)
        println(sub_Vt[i]," ",V_pred[i])
    end
end

function multirestart()
    n_atom, n_basis, g, e = (3, 59, 6, 3)
    r_xy, N, max_deg = (1.4172946, 5, 5)
    idxer = atom_indexer(n_atom)

    H_datat = readdlm("data/h3/h3_data.txt")
    # load atomic coordinates:
    Xt = npzread("data/h3/h3_coord.npy")
    Rt = H_datat[:,1:end-1]; Vt = H_datat[:, end]
    siz = 100
    sub_Rt = Rt[1:siz,:];
    sub_Vt = Vt[1:siz];
    sub_Xt = Xt[1:siz, :, :];
    Θ_vec = rand(Distributions.Uniform(-1.,1.), n_basis*6 + 7)

    restarts = Int(10)
    min_rmse = Inf
    Θ_min = zeros(length(Θ_vec))
    V_pred = f_eval_wrapper_b(Θ_vec, sub_Rt, sub_Xt, n_atom, n_basis, max_deg, idxer, g, e);
    for iter=1:restarts
        # precheck nan:
        while any(isnan.(V_pred)) # reset until no nan:
            println("resetting NaNs!!")
            Θ_vec = rand(Distributions.Uniform(-1.,1.), n_basis*6 + 7)
            """
            res = optimize(Θ -> f_least_squares(f_eval_wrapper_b, sub_Vt, 
                                                Θ_vec, sub_Rt, sub_Xt, n_atom, n_basis, max_deg, idxer, g, e),
                            Θ_vec,
                            Optim.Options(iterations = 1000, show_trace=false);
                            #autodiff = :forward
                            )
            """
            res = LsqFit.curve_fit((R, θ) -> f_eval_wrapper_b(θ, R, sub_Xt, n_atom, n_basis, max_deg, idxer, g, e),
                            sub_Rt, sub_Vt, Θ_vec, show_trace=false, maxIter=2)
            V_pred = f_eval_wrapper_b(res.param, sub_Rt, sub_Xt, n_atom, n_basis, max_deg, idxer, g, e)
        end
        # optimize
        Θ_vec = rand(Distributions.Uniform(-1.,1.), n_basis*6 + 7)
        """
        res = optimize(Θ -> f_least_squares(f_eval_wrapper_b, sub_Vt, 
                                                Θ_vec, sub_Rt, sub_Xt, n_atom, n_basis, max_deg, idxer, g, e),
                            Θ_vec,
                            Optim.Options(iterations = 1000, show_trace=false);
                            #autodiff = :forward
                            )
        """
        res = LsqFit.curve_fit((R, θ) -> f_eval_wrapper_b(θ, R, sub_Xt, n_atom, n_basis, max_deg, idxer, g, e),
                            sub_Rt, sub_Vt, Θ_vec, show_trace=false, maxIter=1000)
        V_pred = f_eval_wrapper_b(res.param, sub_Rt, sub_Xt, n_atom, n_basis, max_deg, idxer, g, e)
        # sort RMSE:
        rmse = f_RMSE(sub_Vt, V_pred)
        println("optimized, restart = ",iter," rmse = ",rmse)
        if rmse < min_rmse
            println("better rmse found!, rmse = ", rmse)
            min_rmse = rmse
            Θ_min = res.param
        end
    end
    writedlm("minimizer_H3_100data.csv", Θ_min)
    x = readdlm("minimizer_H3_100data.csv", '\t')
    V_pred = f_eval_wrapper_b(x, sub_Rt, sub_Xt, n_atom, n_basis, max_deg, idxer, g, e)
    for i=1:length(sub_Vt)
        println(sub_Vt[i]," ",V_pred[i])
    end
    println(min_rmse)
end

#multirestart()

