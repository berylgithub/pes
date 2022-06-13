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

"""
===========
Linear pairpots:
===========
"""


"""
basis functions from Prof. Neumaier
choose basis B inside the function
params:
    - q = R/R_eq, vector n_data
"""
function linratpot_neumbasis(V, R, const_r_xy)
    #n_data = size(R)[1]
    q=R/const_r_xy;
    q1=1 .-q;
    q2=q .^ 2;
    dq=1 .- q2;
    rep1=max.(dq .- q,0).^3 ./ q;         # Coulomb
    rep2=max.(dq,0).^3;              # const+O(q^2)
    rep3=q2.*rep2;                  # O(q^2) = att0
    rep4=q.*rep3;                   # O(q^3)
    att1=q2.*max.(2 .- q,0).^3;         # attractive
    att2=q2.*max.(3 .- q,0).^3;         # attractive
    att3=q2.*max.(4 .- q,0).^3;         # attractive
    att4=max.(q1.*(q .-3),0).^3;       # attractive
    #att5=max.((q .- 2).*(q .- 5),0).^3;       # attractive
    vdw1=max.((1 .- 1 ./ q)./q2,0).^3;   # van der Waals
    vdw2=vdw1 ./ q;                  # van der Waals
    vdw3=vdw2 ./ q;                  # van der Waals
    #vdw4=vdw3 ./ q;                  # van der Waals

    #B = Matrix{Float64}(undef, n_data, 11)
    #B[:, 1] = rep1; 
    B=[rep1,rep2,rep3,rep4,att1,att2,att3,att4,vdw1,vdw2,vdw3];
    #B=[rep1,rep2,rep3,rep4,att1,att2,att3,att4,att5,vdw1,vdw2,vdw3,vdw4];
    #B=[rep1,rep2,rep3,rep4,att1,att2,att3,att4,vdw1,vdw2,vdw3,vdw4];
    #B=[rep1,rep2,rep3,att1,att2,att3,att4,vdw1,vdw2,vdw3];

    B = mapreduce(permutedims, vcat, B) # transform to matrix
    B_T = transpose(B)
    d = q./(q .+ 10);
    DB = diagm(d)*B_T;
    DV = d.*V;
    crep = DB\DV
    Vrep=B_T*crep;
    return Vrep, B, crep, q
end