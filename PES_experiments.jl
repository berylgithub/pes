include("PES_models_bonding_features.jl")
using DataFrames, CSV

"""
this should be where experiments are done, to avoid clutter in ipynb
"""

function ratpot_u_test()
    const_r_xy = 1.4172946
    # test using diatomic:
    H_data = readdlm("data/h2/h2_ground_w.txt")
    R = H_data[:, 1]; V = H_data[:, 2]
    Xs, Ys = shuffleobs((R, V))
    train_data, test_data = splitobs((Xs, Ys); at=0.8)
    R_train = copy(train_data[1]); V_train = copy(train_data[2]);
    R_test = copy(test_data[1]); V_test = copy(test_data[2]);

    # hyperparamteres:
    max_tcheb_deg = 7; e_pow = 1 # 6 may be the best
    ub = 1.; lb = -1.;

    # tuning parameters:
    θ = rand(max_tcheb_deg+1) .* (ub-lb) .+ lb; # random between [a, b] = [-1, 1]

    # precomputations:
    ρ = f_ρ(R_train, const_r_xy)
    q = f_q(ρ)
    p_pol = f_tcheb_u(q, max_tcheb_deg)
    # repeated computations:
    u = f_RATPOT_u(θ, p_pol)
    V_train_tr = V_train .* (ρ .+ (ρ .^ e_pow)) # scale version 1, scale V_train with ρ factors
    ls_val = f_least_squares(f_RATPOT_u, V_train_tr, θ, p_pol)
    println(ls_val)
    # optimize:
    """
    # optim:
    res = optimize(θ -> f_least_squares(f_RATPOT_u, V_train_tr, θ, p_pol),
                    θ, BFGS(),
                    Optim.Options(iterations = 2000, show_trace=true);
                    ) # same result
    """
    res = LsqFit.curve_fit((p_pol, θ) -> f_RATPOT_u(θ, p_pol), p_pol, V_train_tr, θ, show_trace=true, maxIter=500)

    V_pred = f_RATPOT_u(res.param, p_pol)
    for i=1:length(V_pred)
        println(V_train_tr[i]," ",V_pred[i])
    end
    println(f_RMSE(V_pred, V_train_tr))
    # test data:
    ρ = f_ρ(R_test, const_r_xy)
    q = f_q(ρ)
    p_pol = f_tcheb_u(q, max_tcheb_deg)
    u = f_RATPOT_u(θ, p_pol)
    V_test_tr = V_test .* (ρ .+ (ρ .^ e_pow))
    V_pred = f_RATPOT_u(res.param, p_pol)
    println(f_RMSE(V_pred, V_test_tr))
end

"""
enumerate all k = 1:6 × data: [H2, OH+] × method: [RATPOTu, RATPOTu_scale1, RATPOTu_scale2, v_BUMP]
"""
function ratpot_exp()
    # data op:
    #homedir = "/users/baribowo/Code/Python/pes/"
    H_data = readdlm("data/diatomic/h2_ground_w.txt")
    #H_data = readdlm("data/diatomic/oh+_data.txt")
    R = H_data[:, 1]; V = H_data[:, 2]
    Xs, Ys = shuffleobs((R, V))
    train_data, test_data = splitobs((Xs, Ys); at=0.8)
    R_train = copy(train_data[1]); V_train = copy(train_data[2]);
    R_test = copy(test_data[1]); V_test = copy(test_data[2]);

    # hyperparam:
    ## RAT:
    const_r_xy = 1.4172946 # H2
    max_tcheb_deg = 5;
    ## BUMP:
    N = 5

    # tuning param:
    ub = 1.; lb = -1.;
    θ_r = rand(max_tcheb_deg+1) .* (ub-lb) .+ lb; # RAT
    θ_b = rand(2*N+2) .* (ub-lb) .+ lb; # BUMP

    # storage:
    df_train = DataFrame(power=[], RAT=[], RAT1=[], RAT2=[], BUMP=[])
    df_test = DataFrame(power=[], RAT=[], RAT1=[], RAT2=[], BUMP=[])

    # one-time computations, for RAT:
    ρ = f_ρ(R_train, const_r_xy)
    q = f_q(ρ)
    p_pol = f_tcheb_u(q, max_tcheb_deg)
    # one-time comp, BUMP:
    ρ_b = f_ρ(R_train, const_r_xy)
    q_b = f_q_bump(N, ρ_b)
    i_b = f_i(q_b)
    ϵ_b = f_ϵ(i_b, q_b)
    α_b = f_α(ϵ_b)
    β_b = f_β(ϵ_b)

    # repeated computations:
    for e_pow ∈ 1:6
        # RATPOT default:
        res = LsqFit.curve_fit((p_pol, θ) -> v_RATPOT_u(θ, p_pol, ρ, e_pow), p_pol, V_train, θ_r, show_trace=false, maxIter=500)
        V_pred = v_RATPOT_u(res.param, p_pol, ρ, e_pow)
        rmse_r = f_RMSE(V_train, V_pred)
        # RATPOT scale 1:
        V_train_tr = V_train .* (ρ .+ (ρ .^ e_pow))
        res = LsqFit.curve_fit((p_pol, θ) -> f_RATPOT_u(θ, p_pol), p_pol, V_train_tr, θ_r, show_trace=false, maxIter=500)
        V_pred = f_RATPOT_u(res.param, p_pol)
        rmse_r1 = f_RMSE(V_train_tr, V_pred)
        # RATPOT scale 2:
        V_train_tr = V_train .* ρ
        res = LsqFit.curve_fit((p_pol, θ) -> f_RATPOT_u(θ, p_pol, ρ, e_pow), p_pol, V_train_tr, θ_r, show_trace=false, maxIter=500)
        V_pred = f_RATPOT_u(res.param, p_pol, ρ, e_pow)
        rmse_r2 = f_RMSE(V_train_tr, V_pred)
        # BUMP:
        res = LsqFit.curve_fit((ρ_b, θ_b) -> v_BUMP_di(θ_b, ρ_b, q_b, α_b, β_b, i_b, N, e_pow), ρ_b, V_train, θ_b, show_trace=false, maxIter=500)
        V_pred = v_BUMP_di(res.param, ρ_b, q_b, α_b, β_b, i_b, N, e_pow)
        rmse_b = f_RMSE(V_train, V_pred)
        # push result:
        push!(df_train, Dict(:power => e_pow, :RAT => rmse_r, :RAT1 => rmse_r1, :RAT2 => rmse_r2, :BUMP => rmse_b))
    end
    CSV.write("df_train.csv", df_train)
    df_train = CSV.read("df_train.csv", DataFrame)
    println(df_train)
end