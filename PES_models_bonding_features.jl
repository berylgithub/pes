using Optim, LsqFit, NLsolve# opt libs
using LinearAlgebra # self explanatory
using Zygote, ReverseDiff # autodiff
using Plots, LaTeXStrings # plots
using StatsBase, DataStructures, DelimitedFiles, MLUtils, BenchmarkTools, NPZ, StaticArrays # utils
using Distributions # sampling distributions

"""
=== computation utils ===
"""
tsqrt(x) = x > 0. ? √x : √1e-6 # truncated sqrt, sqrt(x) if x > 0 else sqrt(1e-6)
tlog(x) = log(max(0.1, x)) # truncated log 

# use distance to coord implementation in Py for now. Load the files using NPZ

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
=== objectives:
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
=== New bonding features for HxOy: coordinate function and bump functions
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
f_k(t_low, t_hi) = collect(round(t_low):round(t_hi))[2:end-1] # generate a list of k ∈ Z ∩ t_low < k < t_hi


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
    k = f_k(t_low, t_hi); k_idx = convert(Array{Int}, k)
    
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

"""
=== Bonding features:  𝑈,𝑌(𝑧(𝑡),𝑓bump),𝐺(𝑧(𝑡),𝑓bump)
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
        for j=rg_atom
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
        @inbounds rk[:,:,:,i] .= sum(atom_arr, dims=4)
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
    n_elem, n_data, n_k, n_atom = size(rk)
    #rk = permutedims(rk, [1,2,4,3]) # move op to last index
    #G = Array{Float64}(undef, n_data, n_atom, n_k, n_k)
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
    #G = permutedims(G, [1,3,4,2]) # move n_atom to last
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
=== Quadratic models
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
=== Fun evals
"""
function f_pot_bond(Θ, C, R_h, R_C, R_0, 
                    R, X, 
                    r_xy, N, n_atom, n_basis, 
                    idxer, g::Float64=6.0)
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
    return V
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
                Optim.Options(iterations = 1000, show_trace=true);
                )
    # check RMSE:
    V_pred = f_eval_wrapper(res.minimizer, sub_R, sub_X, r_xy, N, n_atom, n_basis, idxer, g)
    println(f_RMSE(sub_V, V_pred))
    for i=1:length(sub_V)
        println(sub_V[i]," ",V_pred[i])
    end
end

