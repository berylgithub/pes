"""
created by Saint8312 23-06-2022
"""

"""
===============
contains all computation utils, such as RMSE, truncated log, etc
===============
"""

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

Maybe useful for opt:
https://discourse.julialang.org/t/fix-parameter-when-passing-to-optimizer/61102

"""

"""
=========================
>>> computation utils <<<
=========================
"""
tsqrt(x) = x > 0. ? √x : √1e-6 # truncated sqrt, sqrt(x) if x > 0 else sqrt(1e-6)
tlog(x) = log(max(0.1, x)) # truncated log

"""
Horner's scheme, given x scalar and C vector of coefficients
"""
function horner(x, C)
    y = C[end]
    len = length(C)
    for i ∈ range(len-1, stop=1, step=-1)
        y = y*x + C[i]
    end
    return y
end

"""
use distance to coord implementation in Py for now. Load the files using NPZ
"""

"""
generates array of coordination indexes for Y[i] vector, which depends on num_atom, e.g.:
num_atom = 3:  [[1,2],[1,3],[2,3]]
                 b_1j  b_2j b_3j
num_atom = 4: [[1,2,3],[1,4,5],[2,4,6],[3,5,6]]
                b_1j     b_2j   b_3j     b_4j
in form of Matrix{Int}(n_atom-1,n_atom), ordered by column (column dominant)
"""
function atom_indexer(n_atom)
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
generator of basis indexes of the tuning parameters
"""
function basis_index_gen(n_basis)
    return [1:n_basis, n_basis+1 : 2*n_basis, 
            2*n_basis+1 : 3*n_basis, 3*n_basis+1 : 4*n_basis,
            4*n_basis+1 : 5*n_basis, 5*n_basis+1 : 6*n_basis]
end


"""
==================
>>> objectives and errors <<<
==================
"""

f_RMSE(X, Y) = √(sum((X .- Y) .^ 2)/length(X))
f_RMSE(ϵ) = √(sum(ϵ.^2)/length(ϵ)) # another dispatch for RMSE
# adjusted RMSE for potential energy, aRMSE^2= RMSE(\delta), \delta=\eps/(|V|+\Delta):
f_ΔV(V, V_l, V_min) = abs.(V) .+ (V_l - V_min) # V_l = V(max(R)), V_min = min(V)
δ_dissociate(V, V_pred, ΔV) = (V .- V_pred) ./ ΔV

"""
dissociated potential: V(θ, R)/(ΔV :=  abs.(V) .+ (V_l - V_min))
"""
function v_dissociate(f_eval, V_l, V_min, f_eval_arg...)
    V = f_eval(f_eval_arg...)
    ΔV = f_ΔV(V, V_l, V_min)
    return V ./ ΔV
end

function f_least_squares(f_eval, Y, f_args...)
    Y_pred = f_eval(f_args...)
    res = sum((Y .- Y_pred).^2)
    return res
end

"""
vector version residual for nonlinear leastsquares solvers
"""
function f_least_squares_vec(f_eval, Y, f_args...)
    Y_pred = f_eval(f_args...)
    res = (Y .- Y_pred).^2
    return res
end


"""
=============
parameter converters, mainly for chebyshev v1 feature, and BUMP v1 feature
==============
"""

"""
convert C, R_h, R_C, R_0, to unconstrained, for old (v1) BUMP
"""
function param_converter(ρ)
    π = Vector{Float64}(undef, length(ρ))
    π[1] = tlog(ρ[1])/20 # log(C)/20
    π[2] = tlog(ρ[2])/20 # log(R_h)/20
    π[4] = tsqrt(ρ[4]) # sqrt(R_0)
    π[3] = tsqrt(ρ[3] - ρ[4]) # sqrt(R_C - R_0)

    return π
end

"""
revert C, R_h, R_C, R_0, to initial, comply to R_h ≤ R_0 ≤ R_C, for old (v1) BUMP
"""
function param_inverter(ρ)
    π = Vector{Float64}(undef, length(ρ))
    π[1] = exp(20. * ρ[1]) # C
    π[2] = exp(20. * ρ[2]) # R_h
    π[4] = ρ[4]^2  # R_0
    π[3] = ρ[3]^2 + π[4] # R_C

    return π
end

"""
convert C, R_h, R_low, R_m, R_0, R_up, R_C, to unconstrained
"""
function param_converter_b(ρ)
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

"""
revert C, R_h, R_low, R_m, R_0, R_up, R_C, to initial, comply to R_h ≤ R_low ≤ R_0 ≤ R_m ≤ R_up ≤ R_C
"""
function param_inverter_b(ρ)
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
