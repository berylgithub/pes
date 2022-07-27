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



