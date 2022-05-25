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
Old RATPOTs (diatomic V)
"""

#=
RATPOT1 for diatomic potential, w/ 3M+1 parameters
params:
    - Θ := training parameters, vector size = 3M+1
    - R := distances, vector, size = n_data
    - Z := nuclear charge, scalar
    - M := max pol power, scalar
=#
function f_ratpot_1(Θ, R, Z, M)
    if Θ[2] < 0 # c₁ > 0
        Θ[2] = -Θ[2]
    end
    # P(R):
    Θ_temp = Θ[4 : 2*M]
    y = map(r -> horner(r, Θ_temp), R)
    y = y .* (R.^2)
    p = Z .* ( (1 ./ R) .+ (Θ[2] .* R) ) .+ Θ[3] .+ y
    # Q(R):
    Θ_temp = Θ[2*M + 1 : 3*M + 1]
    y = map(r -> horner(r, Θ_temp), R)
    q = 1. .+ (y .* R)
    # S(R):
    s = 1. .+ Θ[2] .* ((R .* q).^2)
    # return V:
    return Θ[1] .+ p./s
end

#=
RATPOT2 for diatomic potential, w/ 4M+7 parameters
params:
    - Θ := training parameters, vector size = 4M+7
    - R := distances, vector, size = n_data
    - M := max pol power, scalar
=#
function f_ratpot_2(Θ, R, M)
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

#=
RATPOT3 w/ 4M+8 parameters
params (same as prev ratpots):
    - Θ
    - R
    - Z
    - M
=#
function f_ratpot_3(Θ, R, Z, M)
    a = Θ[1:M]
    b = Θ[M+1:2*M]
    c = Θ[2*M+1:3*M+4]
    d = Θ[3*M+5:4*M+7]
    R0 = Θ[end]

    # turn coefficients to positive:
    # ∀i, bᵢ ≥ 0 :
    bool = b .< 0.
    b[bool] = -b[bool]

    # ∀i, dᵢ ≥ 0:
    bool = d .< 0.
    d[bool] = -d[bool]

    P = Z .* ((1. ./ R) .- (1. ./ R0))
    for i ∈ 1:M
        P .*= (1. .- R ./ a[i]).^2 .+ R ./ b[i]
    end

    Q = (1. .- R ./ c[1]).^2 .+ R ./ d[1]
    for i ∈ 2:M+3
        Q .*= (1. .- R ./ c[i]).^2 .+ R ./ d[i]
    end

    # V = c₀ + P/Q:
    return c[end] .+ (P./Q)
end