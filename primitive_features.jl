"""
created by Saint8312 23-06-2022
"""

include("utils.jl")

"""
===================
>>> Tchebyshev pol primitive features <<<
===================
"""

"""
sub bond strength, similar(R) when using map()
"""
function t_R_fun(R, R_up, R_low, e)
    R2 = R^2
    return ((R2 - R_low^2)/(R_up^2 - R2))^e
end

"""
s_{ij} = s(R_{ij}), similar(R)
t0 = t_R_fun(Rm, R_up, R_low, e)
if R_m = R and R_low < R_m < R_up, then bond_strength_s = 0.5
trainable parameters: (R_low, R_m, R_up)
"""
function s_bond_strength(R, R_up, R_low, t, t0)
    s = 0.
    if R < R_low
        s = 1.
    elseif R_low ≤ R ≤ R_up
        s = t0/(t+t0)
    end
    return s
end

s_dash_f(s) = 2 - 4 * s

"""
bonding pol in tchebyshev term
returns scalar, need to fill deg == 1 manually since it's unreachable
params:
    - out, result storage, containing vector length = deg ∈ Real
    - deg, maximum pol degree, scalar ∈ Int
    - s, scalar ∈ Real
    - s_dash, scalar ∈ Real
"""
function p_tchebyshev_pol(out, deg, s, s_dash)
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

"""
computes the primitive feature in terms of bonding polynomial (tchebyshev pol)
returns array, shape = (n_data, n_d, n_k), where n_k = max_deg
params:
    - R, matrix of distances, shape = (n_data, n_d) ∈ Real
    - R_up, R_m, R_low, tuning params, scalars ∈ Real 
    - max_deg, e, hyperparams, scalars ∈ Real
"""
function f_b_feature(R, R_up, R_m, R_low, max_deg, e)
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
trainable pair potential, all inputs and outputs are scalars, use f.() for array of R !! # somehow this allocates less
reverse AD compatibiliy ??
"""
function V_ref_pairpot(R, C, R_h, R_C, R_0, g)
    V = 0.
    if R ≤ R_h
        V = Inf
    elseif R_h ≤ R ≤ R_C
        R2 = R^2
        V = -C*(R_C^2 - R2)^g * ((R2 - R_0^2)/(R2 - R_h^2))
    end
    return V
end


"""
======================
RATPOTu - ratpot with r_{xy}
======================
"""
f_ρ(R, r_xy) = R./r_xy # computed once
f_q(ρ) = (1. .- ρ)./(1. .+ ρ) # computed once
"""
computed once!!
returns Matrix{Float64}(undef, n_data, max_deg)
params:
    - q, vector, shape = n_data
    - max_deg, scalar ∈ Int
"""
function f_tcheb_u(q, max_deg)
    n_data = length(q)
    # compute tchebyshev polynomial:
    q_dash = map(s_dash_f, q) # vector, length = n_data
    p_pol = Matrix{Float64}(undef, n_data, max_deg) # matrix (n_data, n_k)
    rg_data = 1:n_data
    @simd for i = rg_data
        @inbounds p_tchebyshev_pol((@view p_pol[i,:]), max_deg, q[i], q_dash[i])
    end
    p_pol[:,1] = q
    return p_pol
end

"""
returns u, vector, length = n_data
params:
    - θ, vector, length = max_deg + 1 (extra 1 param for 0th degree of pol)
    - p_pol, matrix, shape = (n_data, max_deg)
!! works for ReverseDiff/Zygote, the problem was mutating arrays, means something like x = Array{Float64}(undef, <sizes>), and changing the values inside the function!!
"""
function f_RATPOT_u(θ, p_pol)
    u = p_pol * θ[2:end] .+ θ[1] #∑p_l(q)θ_l, p degree d, length of θ = d+1
    return u
end

"""
version with scaler as denominator, makes use of multi-dispatcher, observe the param overload!!
addition params:
    - ρ, vector, length = n_data
    - e_pow, scalar
"""
function f_RATPOT_u(θ, p_pol, ρ, e_pow)
    return (p_pol * θ[2:end] .+ θ[1]) ./ (1 .+ (ρ .^ (e_pow - 1)))
end

"""
unalloc ver
"""
function f_RATPOT_u!(u, θ, p_pol)
    u .= p_pol*θ[2:end] .+ θ[1]
end

"""
standard V, directly divided by ρ factors
params:
    - θ, vector, length = max_deg + 1 (extra 1 param for 0th degree of pol)
    - p_pol, matrix, shape = (n_data, max_deg)
    - ρ, similar(f_RATPOT_u)
    - e_pow, scalar ∈ Int
"""
function v_RATPOT_u(θ, p_pol, ρ, e_pow)
    return f_RATPOT_u(θ, p_pol) ./ (ρ .+ (ρ .^ e_pow))
end


"""
================
BUMP feature
================
"""
f_q_bump(N, ρ) = N ./ (1 .+ ρ) # ∈ (0, N], the ρ is from the ratpot function block
f_i(q) = ceil.(q) # ≥ 1
f_ϵ(i, q) = i .- q # ∈ (0, 1]
f_α(ϵ) = (ϵ .* (2 .- ϵ)) .^ 3
f_β(ϵ) = (1 .- (ϵ .^ 2)) .^ 3

# for g exclusive:
f_d(q, k) = q .- k 

# scalar ops:
#f_u_bump(θ, q, α, β, i, N) = (θ[i-1]*q + θ[i+N])*α + (θ[i]*q + θ[i+N+1])*β # u(q), θ here is an element (scalar) of the actual θ vector
f_u_bump(θ, q, α, β, i, N) = (θ[i-1]*(q-i+1) + θ[i+N])*α + (θ[i]*(q-i) + θ[i+N+1])*β
f_w_bump(α, β) = α + β # w(q)

"""
simple scalar ver operation of h_k, all params and output are scalars
"""
function f_h_k(k, i, α, β)
    h = 0.
    if k == i-1
        h = α
    elseif k == i
        h = β
    end
    return h
end

f_h_k(q, k) = abs.((1 .- (q .- k).^2).^3) # another dispatch, vec ver, the fundamental function


"""
computes primitive features,
outputs:
    - u, w, vector, size = n_data
    - h, matrix, size = (n_data, N+1)
params:
    - θ, vector, size = N+1
    - q, α, β, i, vectors, size = n_data
    - N, scalar
"""
function compute_u_w_h_diat!(u, w, h, θ, q, α, β, i, N)
    @simd for el ∈ 1:length(i)
        @inbounds u[el] = f_u_bump(θ, q[el], α[el], β[el], Int(i[el])+1, N) #shift theta by +1, for indexing
        @inbounds w[el] = f_w_bump(α[el], β[el])
    end
    @simd for k ∈ 0:N
        @simd for el ∈ 1:length(i)
            @inbounds h[el, k+1] = f_h_k(k, i[el], α[el], β[el])
        end
    end
end

"""
computes features: h/w and g/w
unrolled loop ver
outputs:
    - x contains h/w, y contains g/w, size(x) = size(y), matrix, size = (N+1, n_data)
params:
    - h, matrix, size = (N+1, n_data)
    - q, vector, size = n_data
    - N, scalar ∈ Int
"""
function compute_hw_gw_diat!(x, y, w, h, q, N)
    n_data = size(q)[1]
    @simd for k ∈ 1:N+1
        @simd for j ∈ 1:n_data
            @inbounds x[j, k] = h[j, k] / w[j] # h./w try vs default
            @inbounds y[j, k] = (q[j] - (k-1)) * x[j, k] # matrix for z:=(q .- (k - 1)) then z.*x
        end
    end
end

"""
computes features: h/w and g/w, v2, where y is computed from g
params:
    - ...
"""
function compute_hw_gw2_diat2!(x, y, h, q, i, N)
    @simd for k ∈ 1:N+1
        @inbounds x[k, :] = (@view h[k, :]) ./ w
        # for g ⟹ y:
        @inbounds d = f_d(q, k-1)
        #@inbounds γ = d.*α
        #@inbounds δ = d.*β
        #@inbounds g = f_h_k.(k-1, i, γ, δ)
        @inbounds y[k, :] = f_h_k.(k-1, i, d.*α, d.*β) ./ w
    end
end

"""
computes V = (u/w) / (ρ + ρ^k) for diatomic
params:
    - u, w, ρ: for diatomic: vectors, length = n_data; for >2atoms: matrix, size = (n_data, n_d) ∈ Float64  
"""
function v_BUMP_di(θ, ρ, q, α, β, i, N, e_pow)
    u = zeros(size(q)[1])
    w = similar(u)
    @simd for el ∈ 1:length(i)
        @inbounds u[el] = f_u_bump(θ, q[el], α[el], β[el], Int(i[el])+1, N) #shift theta by +1, for indexing
        @inbounds w[el] = f_w_bump(α[el], β[el])
    end
    return (u ./ w) ./ (ρ .+ (ρ .^ e_pow))
end

"""
scalar mode hₖ computation (for linear solve)
params:
    - q, k, scalar ∈ Float64
"""
f_h_k_scalar(q, k) = abs((1. - (q - k)^2)^3)

"""
scalar mode w(q):
param:
    - h_k = vector length N+1 ∈ Float64
"""
f_w_q_scalar(h_k) = sum(h_k)


"""
generates matrix A to solve pairpotentials linearly, no alloc!
params:
    - q, vector of data points size = n_data ∈ Float64
    - N, number of bumps, scalar ∈ Float64
outputs:
    - A, matrix, (n_data, 2N+2) ∈ Float64
    - h, matrix, (n_data, N+1) ∈ Float64
    - w, zeros vector, (n_data) ∈ Float64
"""
function BUMP_linear_matrix!(A, h, w, q, N)
    n_data = length(q)
    col_idx = 1:N+1; row_idx = 1:n_data
    # fill h_k:
    @simd for k ∈ col_idx
        @simd for i ∈ row_idx
            h[i, k] = f_h_k_scalar(q[i], k-1) # k-1 due to index starts from 1
        end
    end
    # get w(q) ∀i:
    @simd for k ∈ col_idx
        @simd for i ∈ row_idx
            w[i] += h[i, k]
        end
    end
    # a := h/w and b := (q-k)h/w:
    @simd for k ∈ col_idx
        @simd for i ∈ row_idx
            @inbounds a = h[i, k]/w[i]
            @inbounds b = (q[i] - (k - 1))*a # k-1 due to index starts from 1
            # fill matrix A:
            A[i, k] = a
            A[i, k+N+1] = b
        end
    end
end

"""
same as above but BUMPs only, no wavelets
difference:
    - A, matrix, (n_data, N+1) ∈ Float64
"""
function BUMP_only!(A, h, w, q, N)
    n_data = length(q)
    col_idx = 1:N+1; row_idx = 1:n_data
    # fill h_k:
    @simd for k ∈ col_idx
        @simd for i ∈ row_idx
            h[i, k] = f_h_k_scalar(q[i], k-1) # k-1 due to index starts from 1
        end
    end
    # get w(q) ∀i:
    @simd for k ∈ col_idx
        @simd for i ∈ row_idx
            w[i] += h[i, k]
        end
    end
    # a := h/w and b := (q-k)h/w:
    @simd for k ∈ col_idx
        @simd for i ∈ row_idx
            @inbounds a = h[i, k]/w[i]
            # fill matrix A:
            A[i, k] = a
        end
    end
end

"""
computes primitive features, for atom > 2
unrolled version, C++ like syntax and speed
outputs:
    - u, w, vector, size = (n_data, n_d)
    - h, array, size = (n_data, n_d, N+1)
params:
    - θ, vector, size = N+1
    - q, α, β, i, similar(u)
    - N, scalar
"""
function compute_u_w_h!(u, w, h, θ, q, α, β, i, N)
    n_data, n_d = size(q)
    @simd for m ∈ 1:n_d
        @simd for el ∈ 1:n_data
            @inbounds u[el, m] = f_u_bump(θ, q[el, m], α[el, m], β[el, m], Int(i[el, m])+1, N) # shift theta by +1, for indexing
            @inbounds w[el, m] = f_w_bump(α[el, m], β[el, m])
        end
    end
    @simd for k ∈ 0:N
        @simd for m ∈ 1:n_d
            @simd for el ∈ 1:n_data
                @inbounds h[el, m, k+1] = f_h_k(k, i[el, m], α[el, m], β[el, m])
            end
        end
    end
end

"""
computes features: h/w and g/w, for atom > 2
unrolled loop ver
outputs:
    - x contains h/w, y contains g/w, size(x) = size(y), array, size = (n_data, n_d, N+1)
params:
    - h, array, size = (n_data, n_d, N+1)
    - w, matrix, size = (n_data, n_d)
    - q, matrix, size = (n_data, n_d)
    - N, scalar ∈ Int
"""
function compute_hw_gw!(x, y, w, h, q, N)
    n_data, n_d = size(q)
    @simd for k ∈ 1:N+1
        @simd for d ∈ 1:n_d
            @simd for j ∈ 1:n_data
                @inbounds x[j, d, k] = h[j, d, k] / w[j, d] # h./w try vs default
                @inbounds y[j, d, k] = (q[j, d] - (k-1)) * x[j, d, k] # matrix for z:=(q .- (k - 1)) then z.*x
            end
        end
    end
end

"""
main block computation for the new LC of bump features
params:
    - θ, vector, size = 2N+2, TUNING PARAM!!
    - R, matrix of distances, size = (n_data, n_d)
    - const, N, selfexplanatoryt
returns:
    - ρ,u,w, matrix similar(R)
    - x,y, array size = (n_data, n_d, N+1)
"""
function BUMP_feature(θ, R, const_r_xy, N)
    ρ = f_ρ(R, const_r_xy) # used for U
    q = f_q_bump(N, ρ)
    i = f_i(q)
    ϵ = f_ϵ(i, q)
    α = f_α(ϵ)
    β = f_β(ϵ)

    # compute primitives, u,w,h, (u, w) is used for U:
    n_data, n_d = size(R)
    u = similar(R)
    w = similar(u) 
    h = zeros(n_data, n_d, N+1) # stores the main sub primitive
    compute_u_w_h!(u, w, h, θ, q, α, β, i, N)

    # compute x := h/w, y := g/w, in place of Tchebyshev
    x = similar(h)
    y = similar(x)
    compute_hw_gw!(x, y, w, h, q, N)
    
    return ρ,u,w,x,y
end

"""
custom concatenation of BUMP feature, roughly the array should contain [x1,y1,x2,y2,...,xmax_deg or ymaxdeg] in the n_k dim
"""
function concat_BUMP(x, y, max_deg)
    n_data, n_d, _ = size(x)
    b = Array{Float64}(undef, n_data, n_d, max_deg)
    xcount = ycount = 1
    @simd for i=1:max_deg # put x in odd, y in even index:
        if i%2 == 0
            @inbounds b[:,:,i] = view(y,:,:,ycount)
            ycount += 1
        else
            @inbounds b[:,:,i] = view(x,:,:,xcount)
            xcount += 1
        end
    end
    return b
end


"""
============
Chebyshev from q (scaled R) features
============
"""

"""
linear Vref in chebyshev term, this is a precomputation!
params:
    - θ, parameters from solving linear ratpot, vector, size = n_param
    - R, matrix of distances, size = (n_data, n_d)
outputs:
    - Vref, matrix, size = (n_data, n_d) ∈ Float64
"""
function chebq_vref(θ, p, n_data, n_d)
    vref = Matrix{Float64}(undef, n_data, n_d)
    @simd for i ∈ 1:n_d 
        @inbounds vref[:, i] = (@view p[:,:,i])*θ
    end
    return vref
end


"""
feature ≡ b in terms of the old chebyshev feature. Returns matrix size = (n_data, n_d, d)
takes precomputed ρ and q array. 
"""
function chebq_feature(ρ, q, d, n_data, n_d)
    p = Array{Float64}(undef, n_data, d+1, n_d)
    vec_ones = ones(n_data) # only for the Vref
    @simd for i ∈ 1:n_d
        @inbounds p[:, 1, i] = vec_ones # only for the Vref
        @inbounds p[:, 2:end, i] = f_tcheb_u((@view q[:, i]), d) ./ (2*@view ρ[:, i]) # the default formula: denom ≈ (ρ .+ ρ)^k, since k =1, this is chosen
    end
    return p
end


"""
other dispatch, for U basis
params: 
    - u, w, ρ, matrix = similar(R)
    - e_pow, scalar
"""
function v_BUMP_di(u, w, ρ, e_pow)
    return (u ./ w) ./ (ρ .+ (ρ .^ e_pow))
end