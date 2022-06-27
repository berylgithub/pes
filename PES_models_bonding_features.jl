using Optim, LsqFit, NLsolve # opt libs
using LinearAlgebra # self explanatory
using Zygote, ReverseDiff, ForwardDiff # autodiff
#using Enzyme # not yet used
using Plots, LaTeXStrings # plots
using StatsBase, DataStructures, MLUtils # data processing utils
using Distributions # sampling distributions
using DelimitedFiles, JSON3, NPZ # file utils
using BenchmarkTools # selfexplanatory

# includes:
include("utils.jl")
include("primitive_features.jl")

"""
======================
>>> Bonding features:  𝑈,𝑌(𝑧(𝑡),𝑓bump),𝐺(𝑧(𝑡),𝑓bump) <<<
======================
"""

"""
Y_d[i] = sum_{j neq i}b_{ijd}
returns an array with shape = (n_data, n_k, n_atom)
params:
    - z_bump_mat, array containing z(t) and bump functions, shape = (n_data, n_d, n_k) ∈ Float64
    - idxer, matrix containing the indexer required for 2-body sums, shape = (n_atom-1, n_atom)
"""
function f_Y_coord(z_bump_mat, idxer)
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

"""
X_j - X_i, i \neq j
returns: Δ array containing the differences of the coordinates, shape = (n_data, n_elem=3, n_d) ∈ Float64
params:
- X, matrix containing coordinates of the atoms, shape = (n_data, n_atom, 3)
"""
function f_Δcoord(X)
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
"""
r_d[i] ∈ R^3 = sum(z_bump_ij*Δ_ij) -> scalar*vector
returns: array of r, shape = (3, n_data, n_k, n_atom)
params:
    - z_bump_mat, array containing z(t) and bump functions, shape = (n_data, n_d, n_k) ∈ Float64
    - Δ, array containing the differences of the coordinates, shape = (n_data, 3, n_d) ∈ Float64
    - idxer, matrix containing the atomic indexer, shape = (n_atom-1,n_atom) ∈ Int
"""
function f_r_orient_vec(z_bump, Δ, idxer)
    n_data, n_d, n_k = size(z_bump); n_atom = size(idxer)[2]
    temp_out = Array{Float64}(undef, 3, n_data, n_d, n_k)
    #temp_out = @SArray zeros(n_data, n_d, n_k)
    rg_k = 1:n_k; rg_j = 1:n_d; rg_i=1:n_data
    # vanilla loop for z[i,j,k]*Δ[i,:,j]:
    Δ = permutedims(Δ, [2,1,3]) # put vector length to front
    @simd for k=rg_k
        @simd for j=rg_j
            @simd for i=rg_i
                @simd for h ∈ 1:3
                    @inbounds temp_out[h,i,j,k] = z_bump[i,j,k]*Δ[h,i,j] # need to change this, this allocates heavily!!
                end
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

"""
G[i]_k1k2 ∈ R = r_k1[i] ⋅ r_k2[i]
returns:
    - G, array from dot product of rk, shape = (n_data, n_k, n_k, n_atom)
params:
    - rk, array of orientation vectors, shape = (3, n_data, n_k, n_atom)
"""
function f_G_mat(rk)
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

"""
U = ∑V_ij
returns matrix (n_data, n_atom) ∈ Float64
params:
    - R, matrix of distances, shape = (n_data, n_d) ∈ Float64
    - idxer, matrix containing the atomic indexer, shape = (n_atom-1,n_atom) ∈ Int
    - all params of V_ref_pairpot (arg_vref...)
"""
function f_U_bas(R, idxer, arg_vref...)
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

"""
U = ∑V_ij, for BUMP features
returns matrix (n_data, n_atom) ∈ Float64
params:
    - idxer, matrix containing the atomic indexer, shape = (n_atom-1,n_atom) ∈ Int
    - all params of the pair potentials (arg_vref...), most important ones: u,w, size = similar(R)
"""
function f_U_bas_BUMP(idxer, arg_vref...)
    n_data, _ = size(arg_vref[1]) # which is u
    n_atom = size(idxer)[2]
    Vref = v_BUMP_di(arg_vref...)
    U = Matrix{Float64}(undef, n_data, n_atom)
    @simd for i=1:n_atom
        Vsub = @view Vref[:, idxer[:,i]]
        U[:, i] = sum(Vsub, dims=2)
    end
    U = U./maximum(abs.(U)) # scale U, by U:=U/max(abs(U))
    return U
end

"""
more general U, Vref as an arbitrary fun, Vref should output a matrix, size = (n_data, n_d)
additional params:
    - f_Vref, function that computes Vref which returns matrix w/ size = (n_data, n_d)
"""
function f_U_bas_general(idxer, f_Vref, arg_vref...)
    n_data = arg_vref[end-1] # n_data, n_d = end-1, end
    n_atom = size(idxer)[2]
    Vref = f_Vref(arg_vref...)
    U = Matrix{Float64}(undef, n_data, n_atom)
    @simd for i=1:n_atom
        Vsub = @view Vref[:, idxer[:,i]]
        U[:, i] = sum(Vsub, dims=2)
    end
    #U = U./maximum(abs.(U)) # scale U, by U:=U/max(abs(U))
    return U
end


"""
constructs Φ array of basis (mathematically, a matrix), all in and outputs' arrays ∈ Float64
returns: array with shape = (n_data, n_basis, n_atom)
params:
    - U, array, shape = (n_data, n_atom)
    - Y, array, shape = (n_data, n_k, n_atom)
    - G, array, shape = (n_data, n_k, n_k, n_atom)
    - n_basis, number of basis, scalar ∈ Int
"""
function f_Φ(U, Y, G, n_basis)
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

"""
rational quadratic model of A form
returns a vector, shape = n_data ∈ Float64
params:
    - θ ⊂ Θ, subset of the parameter matrix, shape = (n_basis, 2) ∈ Float64
    - ϕ:=Φ[i] ⊂ Φ, subset of the basis array (indexed by atom), shape = (n_data, n_basis) ∈ Float64
"""
function f_A(θ, ϕ)
    # using matrix*vector mult:
    numer = ϕ * (@view θ[:,1])
    denom = ϕ * (@view θ[:,2])
    denom = denom.^2 .+ 1.
    return numer ./ denom
end

"""
rational quadratic model of T0 form, numer^2
returns a vector, shape = n_data ∈ Float64
params:
    - θ ⊂ Θ, subset of the parameter matrix, shape = (n_basis, 2) ∈ Float64
    - ϕ:=Φ[i] ⊂ Φ, subset of the basis array (indexed by atom), shape = (n_data, n_basis) ∈ Float64
"""
function f_T0(θ, ϕ)
    numer = ϕ * (@view θ[:,1])
    denom = ϕ * (@view θ[:,2])
    denom = denom.^2 .+ 1.
    return (numer).^2 ./ denom
end

"""
the sum of atomic energy terms. ϵ = ∑ϵ0[i], where ϵ0[i] := A[i] - √(B[i] + C[i])
returns a vector, shape = n_data ∈ Float64
params:
    - Θ, tuning parameter matrix, shape = (n_basis, 6) ∈ Float64
    - Φ, basis array, shape = (n_data, n_basis, n_atom) ∈ Float64
"""
function f_energy(Θ, Φ)
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
works for AD !!!
"""
function f_energy_AD(Θ, Φ)
    n_data, n_basis, n_atom = size(Φ)
    ϵ = zeros(n_data)
    for i ∈ 1:n_atom # for each atom:
        A = f_A((@view Θ[:,1:2]), (@view Φ[:,:,i])) # A term
        B = f_T0((@view Θ[:,3:4]), (@view Φ[:,:,i])) # B term
        C = f_T0((@view Θ[:,5:6]), (@view Φ[:,:,i])) # C term
        ϵ0 = A .- .√(B .+ C) # |vector| = n_data
        ϵ += ϵ0
    end
    return ϵ
end

"""
single primitive OPs:
"""
function f_A_single(θ_u, θ_l, ϕ)
    # using matrix*vector mult:
    numer = sum(ϕ .* θ_u)
    denom = sum(ϕ .* θ_l)
    denom = denom^2 + 1.
    return numer / denom
end

function f_T0_single(θ_u, θ_l, ϕ)
    numer = sum(ϕ .* θ_u)
    denom = sum(ϕ .* θ_l)
    denom = denom^2 + 1.
    return (numer)^2 / denom
end

"""
single data f_energy
params:
    - Θ, vector (not matrix!!), (n_basis*6)
    - ϕ ⊂ Φ, matrix (n_atom, n_basis)
"""
function f_energy_single(Θ, ϕ, n_atom)
    # all operations in scalar:
    ϵ = 0.
    for i ∈ 1:n_atom
        A = f_A_single(Θ[1:n_basis], Θ[n_basis+1 : 2*n_basis], (@view ϕ[:,i]))
        B = f_T0_single(Θ[2*n_basis+1 : 3*n_basis], Θ[3*n_basis+1 : 4*n_basis], (@view ϕ[:,i]))
        C = f_T0_single(Θ[4*n_basis+1 : 5*n_basis], Θ[5*n_basis+1 : 6*n_basis], (@view ϕ[:,i]))
        ϵ0 = A - √(B + C)
        ϵ += ϵ0
    end
    return ϵ
end

"""
===============
>>> Main fun evals for z and bumps <<<
===============
"""

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
function f_pot_bond(Θ, C, R_h, R_C, R_0, 
        R, X, 
        r_xy, N, n_atom, n_basis, 
        idxer, g=6)

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


"""
wrapper for function evaluation, so that the tuning param is a long vector
returns: V, vector of energy, shape = n_data
params:
- Θ_vec, tuning parameters, vector, shape = n_basis*6 + 4
"""
function f_eval_wrapper(Θ_vec, arg_f...)
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
function f_pot_bond_b(Θ, C, R_h, R_low, R_0, R_m, R_up, R_C, 
        R, X, 
        n_atom, n_basis, max_deg,
        idxer, g=6, e=3)
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

"""
wrapper for function evaluation, so that the tuning param is a long vector
returns: V, vector of energy, shape = n_data
params:
- Θ_vec, tuning parameters, vector, shape = n_basis*6 + 7
"""
function f_eval_wrapper_b(Θ_vec, arg_f...)
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
================================
>>> Main evals for BUMP
================================
"""

"""
function evaluation of V(.) using bonding features from BUMP.
    returns V, vector of potential energy, shape = n_data ∈ Real
    params:
        - θ, tuning param, vector, length = 2N+2
        - Θ, tuning param, matrix, size = (n_basis, 6)
        - ...
"""
function f_pot_bond_BUMP(θ, Θ, R, X, idxer, const_r_xy, n_basis, N, e_pow, max_deg)
    ρ,u,w,x,y = BUMP_feature(θ, R, const_r_xy, N);
    # U basis:
    U = f_U_bas_BUMP(idxer, u, w, ρ, e_pow)
    # b subfeature concatenation:
    b = concat_BUMP(x, y, max_deg)
    # Y basis:
    Y = f_Y_coord(b, idxer)
    Y = Y ./ maximum(abs.(Y)) # scaler
    # G basis:
    Δ = f_Δcoord(X)
    rk = f_r_orient_vec(b, Δ, idxer)
    rk = rk ./ maximum(abs.(rk)) # scaler
    G = f_G_mat(rk)
    # Φ basis:
    Φ = f_Φ(U, Y, G, n_basis)
    # compute total energy:
    V = f_energy(Θ, Φ)
    return V
end

"""
wrapper for BUMP feval, takes in a vector of parameters (θ_vec)
"""
function f_eval_wrapper_BUMP(Θ_vec, arg_f...)
    # unroll coefficients:
    n_basis, N = arg_f[[5, 6]]
    Θ = Matrix{Float64}(undef, n_basis, 6)
    for i ∈ 1:6
        Θ[:, i] = Θ_vec[((i-1)*n_basis) + 1 : i*n_basis]
    end
    θ = Θ_vec[end - (2*N+1) : end] # pairpot params
    V = f_pot_bond_BUMP(θ, Θ, arg_f...)
    return vec(V)
end

"""
================================
>>> Main evals for linear RATPOTs
================================
"""

"""
(Pre computation mode !!)
computes the basis functions Φ using bonding features from linear RATPOTs.
    returns V, vector of potential energy, shape = n_data ∈ Float64
    params:
        - R (n_data, n_d) ∈ Float64
        - X (H_coord) ...
        - θ, optimized parameter from linear ratpots, (d) ∈ Float64
        - indexer
        - ...
"""
function f_pot_pre(R, H_coord, θ, indexer,
                    const_r_xy, d, max_d, n_basis,
                    n_data, n_d)
    ρ = f_ρ(R, const_r_xy)
    q = f_q(ρ)
    p = chebq_feature(ρ, q, d, n_data, n_d)
    U = f_U_bas_general(indexer, chebq_vref, θ, p, n_data, n_d)
    p = permutedims(p, [1,3,2]) # becomes (n_data, n_d, d+1)
    p = @view p[:, :, 2:1+max_d] # take only the relevant index
    Y = f_Y_coord(p, indexer)
    Δ = f_Δcoord(H_coord)
    rk = f_r_orient_vec(p, Δ, indexer)
    G = f_G_mat(rk)
    Φ = f_Φ(U, Y, G, n_basis)
    return Φ
end

"""
feval wrapper s.t. it accepts vector as tuning parameters instead of matrix or array.
"""
function f_energy_wrap(θ, Φ, n_basis)
    Θ = Matrix{Float64}(undef, n_basis, 6)
    for i ∈ 1:6
        Θ[:, i] = θ[((i-1)*n_basis) + 1 : i*n_basis]
    end
    return vec(f_energy(Θ, Φ))
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
                            sub_Rt, sub_Vt, Θ_vec, show_trace=false, maxIter=100)

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

function basis_precomp_opt_test()
    homedir = "/users/baribowo/Code/Python/pes/" # for work PC only, julia bash isn't available.
    #homedir = "" # default
    # compute optimal ratpot:
    data = readdlm(homedir*"data/h2/h2_ground_w.txt")
    R = data[:, 1]; V = data[:, 2]
    siz = size(R)[1]
    id_train = []; id_test = []
    for i ∈ 1:siz
        if i % 2 == 1
            push!(id_train, i)
        elseif i % 2 == 0
            push!(id_test, i)
        end
    end
    R_train = R[id_train]; R_test = R[id_test]
    V_train = V[id_train]; V_test = V[id_test]
    # hyperparam for linratpot:
    const_r_xy = 1.4172946
    V_min = minimum(V)
    V_l = V[argmax(R)]
    Δ = V_l - V_min

    d = 18 # best d from experiment
    θ, A, q = linratpot_cheb(V, R, const_r_xy, d, 1)
    V_pred = A*θ
    rmse = f_RMSE(V, V_pred)
    armse = Δ*f_RMSE(δ_dissociate(V, V_pred, f_ΔV(V_pred, V_l, V_min)))
    println(rmse, armse)

    # load molecule data:
    H_data = readdlm(homedir*"data/h3/h3_data.txt")
    # load atomic coordinates:
    X = npzread(homedir*"data/h3/h3_coord.npy")
    R = H_data[:,1:end-1]; V = H_data[:, end]
    n_data = size(R)[1]
    siz = 100
    sub_R = R[1:siz,:];
    sub_V = V[1:siz];
    sub_X = X[1:siz, :, :];

    # hyperparams for feval:
    max_d = 5; n_basis = 59
    n_data, n_d = size(sub_R)
    ub = 1.; lb = -1.

    # precompute basis!!:
    Φ = f_pot_pre(sub_R, sub_X, θ, atom_indexer(3), const_r_xy, d, max_d, n_basis, n_data, n_d)

    # optimize:
    Θ = rand(n_basis*6).* (ub-lb) .+ lb # tuning parameter
    res = LsqFit.curve_fit((Φ, Θ) -> f_energy_wrap(Θ, Φ, n_basis),
                            Φ, sub_V, Θ, show_trace=false, maxIter=1000)

    V_pred = f_energy_wrap(res.param, Φ, n_basis)
    for i=1:length(sub_V)
        println(sub_V[i]," ",V_pred[i])
    end
    println(f_RMSE(sub_V, V_pred))
end

