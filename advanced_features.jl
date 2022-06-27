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