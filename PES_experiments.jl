include("PES_models_bonding_features.jl")
using DataFrames, CSV

"""
this should be where experiments are done, to avoid clutter in ipynb
"""

"""
enumerate all k = 1:6 × data: [H2, OH+] × method: [RATPOTu, RATPOTu_scale1, RATPOTu_scale2, v_BUMP]
"""
function ratpot_exp()
    # data op:
    #homedir = "/users/baribowo/Code/Python/pes/"
    #H_data = readdlm("data/diatomic/h2_ground_w.txt")
    H_data = readdlm("data/diatomic/oh+_data.txt")
    R = H_data[:, 1]; V = H_data[:, 2]
    Xs, Ys = shuffleobs((R, V))
    train_data, test_data = splitobs((Xs, Ys); at=0.8)
    R_train = copy(train_data[1]); V_train = copy(train_data[2]);
    R_test = copy(test_data[1]); V_test = copy(test_data[2]);
    
    # hyperparam:
    ## RAT:
    #const_r_xy = 1.4172946 # H2
    const_r_xy = 1.9369 #OH+
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
    CSV.write("df_train_OH+.csv", df_train)
    df_train = CSV.read("df_train_OH+.csv", DataFrame)
    println(df_train)
end

function multirestart_BUMP()
    # hyperparam:
    n_atom, n_basis, e_pow = (3, 59, 3)
    const_r_xy, N, max_deg = (1.4172946, 2, 5)
    idxer = atom_indexer(n_atom)

    # data loader:
    homedir = "/users/baribowo/Code/Python/pes/"
    H_data = readdlm(homedir*"data/h3/h3_data.txt") # potential data
    X = npzread(homedir*"data/h3/h3_coord.npy") # atomic coordinates
    R = H_data[:,1:end-1]; V = H_data[:, end]
    siz = size(R)[1]
    sub_R = R[1:siz,:];
    sub_V = V[1:siz];
    sub_X = X[1:siz, :, :];
    println("data size =",siz)

    # data split:
    idxes = shuffleobs(1:siz) # shuffle indexes
    id_train, id_test = splitobs(idxes, at=0.8) # split train and test indexes
    # split data by index:
    R_train = sub_R[id_train,:]; V_train = sub_V[id_train];
    R_test = sub_R[id_test,:]; V_test = sub_V[id_test]
    X_train = sub_X[id_train,:,:]; X_test = sub_X[id_test,:,:]
    println("train size = ",length(V_train))
    println("test size = ", length(V_test))

    #tuning param:
    len_param = n_basis*6 + 2*N + 2
    Θ_vec = rand(Distributions.Uniform(-1.,1.), len_param)

    restarts = Int(100) # number of restarts for the multirestart method
    min_rmse = Inf
    Θ_min = zeros(length(Θ_vec))
    V_pred = f_eval_wrapper_BUMP(Θ_vec, R_train, X_train, idxer, const_r_xy, n_basis, N, e_pow, max_deg)

    t = @elapsed begin # timer
        for iter=1:restarts
            # precheck nan:
            while any(isnan.(V_pred)) # reset until no nan:
                println("resetting NaNs!!")
                Θ_vec = rand(Distributions.Uniform(-1.,1.), len_param)
                res = LsqFit.curve_fit((R, θ) -> f_eval_wrapper_BUMP(θ, R, X_train, idxer, const_r_xy, n_basis, N, e_pow, max_deg),
                                R_train, V_train, Θ_vec, show_trace=false, maxIter=2)
                V_pred = f_eval_wrapper_BUMP(res.param, R_train, X_train, idxer, const_r_xy, n_basis, N, e_pow, max_deg)
            end
            # optimize
            Θ_vec = rand(Distributions.Uniform(-1.,1.), len_param)
            res = LsqFit.curve_fit((R, θ) -> f_eval_wrapper_BUMP(θ, R, X_train, idxer, const_r_xy, n_basis, N, e_pow, max_deg), 
                            R_train, V_train, Θ_vec, show_trace=false, maxIter=1000)
            V_pred = f_eval_wrapper_BUMP(res.param, R_train, X_train, idxer, const_r_xy, n_basis, N, e_pow, max_deg)
            # sort RMSE:
            rmse = f_RMSE(V_train, V_pred)
            println("optimized, restart = ",iter," rmse = ",rmse)
            if rmse < min_rmse
                println("better rmse found!, rmse = ", rmse)
                min_rmse = rmse
                Θ_min = res.param
            end
        end
    end # end of timer
    writedlm(homedir*"minimizer_H3.csv", Θ_min)
    x = readdlm(homedir*"minimizer_H3.csv", '\t')
    # test data evaluation:
    V_pred = f_eval_wrapper_BUMP(x, R_test, X_test, idxer, const_r_xy, n_basis, N, e_pow, max_deg)
    for i=1:length(V_test)
        println(V_test[i]," ",V_pred[i])
    end
    println("min train RMSE = ",min_rmse)
    println("test RMSE = ", f_RMSE(V_pred, V_test))
    println("elapsed multirestart time = ",t)
end
