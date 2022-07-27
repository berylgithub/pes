using JLD

"""
params:
    - m size of subsets
    - n size of of molecule (num atoms)
    - file, string, directory
    - mode, string = "Z3", "D3"
"""
function check_Z3(x)
    return sum(x .^ 2) ≤ 4
end

function check_D3(x)
    return sum(x .^ 2) ≤ 8
end

function generate_subsets(m, n, file, mode="Z3")
    idx = 1
    Γ = Array{Int64}(undef, 3, n, m)
    while idx ≤ m
        T = rand(-2:2, (3,n))
        accept = 0
        for i=1:n
            if mode == "Z3" 
                accept += check_Z3(T[:,i])
            elseif mode == "D3"
                accept += check_D3(T[:,i])
            end
        end
        if accept == n
            Γ[:, :, idx] = T
            idx += 1
        end
    end
    JLD.save(file, "data", Γ)
    return Γ
end


"""
generate a symmetric matrix from a matrix of coordinates, 
each entry is a squared distance.
params:
    * R, output, symmetric matrix of distances, zeros = (n_atom, n_atom)
    - T, matrix of coordinates, size = (3, n_atom)
    - n_atom, num of atoms, Int64
"""
function compute_distances!(R, T, n_atom)
    for i=1:n_atom
        for j=1:n_atom 
            if i<j
                R[i,j] = sum((T[:,i] .- T[:,j]).^2)
            elseif i > j
                R[i,j] = R[j,i]
            end
        end
    end
end

"""
computes the distances from the array of coordinates.
params:
    - Γ, array of coordinates, size = (3, n_atom, m).
outputs:
    - Γ_dist, array of symmetric distance matrix, size = (n_atom, n_atom, m)
"""
function transform_to_distances(Γ)
    _, n, m = size(Γ)
    Γ_dist = zeros(Int64, n, n, m)
    R = zeros(Int64, 4,4)
    for i ∈ 1:m
        compute_distances!(R, Γ[:, :, i], n)
        Γ_dist[:, :, i] = R
    end
    return Γ_dist
end

function compute_features(Γ)
    
end


function check_isomorphism(Tʲ, Tᵏ)
        
end
