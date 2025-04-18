
include("utils.jl")
include("primitive_features.jl")
include("advanced_features.jl")
include("PES_models_bonding_features.jl")
include("RATPOT.jl")

using DataFrames, CSV

"""
this should be where experiments are done, to avoid clutter in ipynb
"""

"""
enumerate all k = 1:6 × data: [H2, OH+] × method: [RATPOTu, RATPOTu_scale1, RATPOTu_scale2, v_BUMP]
"""
function diatomic_bump_opt()
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

"""
recompute coeffs of rat1 rat2
"""
function ratpots_opt()
    # load data:
    homedir = "/users/baribowo/Code/Python/pes/" # for work PC only, julia bash isn't available.
    #homedir = "" 
    H_data = readdlm(homedir*"data/h2/h2_ground_w.txt")
    R = H_data[:, 1]; V = H_data[:, 2]
    id_train = vec(readdlm(homedir*"data/diatomic/index_train_H2.csv", Int))
    id_test = vec(readdlm(homedir*"data/diatomic/index_test_H2.csv", Int))
    R_train = R[id_train,:]; V_train = V[id_train];
    R_test = R[id_test,:]; V_test = V[id_test];

    # storage:
    df = DataFrame(M=[], RAT1=[], RAT2=[], RAT3=[])
    df_param = DataFrame(M=[], RAT1_θ = [], RAT2_θ =[], RAT3_θ = []) 

    # for each ratpot, do several restarts:
    list_M = [5, 7, 10, 13] # pick from the best M later for time
    
    # hyperparam:
    Z = 1;
    # multirestart param:
    restarts = 30
    ub = 1.; lb = -1.
    for M ∈ list_M
        # multirestart:
        println("M = ",M)
        min_rmse_1 = min_rmse_2 = min_rmse_3 = Inf
        θ_min_1 = zeros(3*M+1)
        θ_min_2 = zeros(4*M+7)
        θ_min_3 = zeros(4*M+8)
        for i ∈ 1:restarts
            # ratpot1:
            θ = rand(3*M+1) .* (ub-lb) .+ lb
            res = optimize(θ -> f_least_squares(f_ratpot_1, V_train, θ, R_train, Z, M),
                    θ, BFGS(),
                    Optim.Options(iterations = 2000, show_trace=false); 
                    autodiff = :forward
                    )
            V_pred = f_ratpot_1(res.minimizer, R_train, Z, M)
            rmse = f_RMSE(V_pred, V_train)
            println("rat1, restart = ",i, " rmse = ",rmse)
            if rmse < min_rmse_1
                println("better rmse found!!")
                min_rmse_1 = rmse
                θ_min_1 = res.minimizer
            end

            # ratpot2:
            θ = rand(4*M+7) .* (ub-lb) .+ lb
            res = optimize(θ -> f_least_squares(f_ratpot_2, V_train, θ, R_train, M),
                    θ, BFGS(),
                    Optim.Options(iterations = 2000, show_trace=false); 
                    autodiff = :forward
                    )
            V_pred = f_ratpot_2(res.minimizer, R_train, M)
            rmse = f_RMSE(V_pred, V_train)
            println("rat2, restart = ",i, " rmse = ",rmse)
            if rmse < min_rmse_2
                println("better rmse found!!")
                min_rmse_2 = rmse
                θ_min_2 = res.minimizer
            end

            # ratpot2:
            θ = rand(4*M+8) .* (ub-lb) .+ lb
            res = optimize(θ -> f_least_squares(f_ratpot_3, V_train, θ, R_train, Z, M),
                    θ, BFGS(),
                    Optim.Options(iterations = 2000, show_trace=false); 
                    autodiff = :forward
                    )
            V_pred = f_ratpot_3(res.minimizer, R_train, Z, M)
            rmse = f_RMSE(V_pred, V_train)
            println("rat3, restart = ",i, " rmse = ",rmse)
            if rmse < min_rmse_3
                println("better rmse found!!")
                min_rmse_3 = rmse
                θ_min_3 = res.minimizer
            end
        end
        push!(df, Dict(:M => M, :RAT1 => min_rmse_1, :RAT2 => min_rmse_2, :RAT3 => min_rmse_3))
        push!(df_param, Dict(:M => M, :RAT1_θ => θ_min_1, :RAT2_θ => θ_min_2, :RAT3_θ => θ_min_3))
    end
    CSV.write(homedir*"df_rat.csv", df)
    df = CSV.read(homedir*"df_rat.csv", DataFrame)
    println(df)

    CSV.write(homedir*"df_param_rat.csv", df_param)
    df_param = CSV.read(homedir*"df_param_rat.csv", DataFrame);
end

function linear_ratpots()
    homedir = "/users/baribowo/Code/Python/pes/"
    # storage:
    #df_rmse = DataFrame(deg=[], neum=[], cheb=[], bump=[])

    data_list = ["H2", "O2", "OH"]
    file_list = ["data/h2/h2_ground_w.txt", "data/diatomic/o2_data.txt", "data/diatomic/oh_data.txt"]
    io = open(homedir*"linear_pairpot_scaled.txt", "w")
    scaling = range(10., 200., step=10.) # scaling
    for file_idx ∈ 1:length(file_list)
        println("data = ",data_list[file_idx])
        write(io, "data = ",string(data_list[file_idx]), "\n")
        # load data and split 50:50:
        data = readdlm(homedir*file_list[file_idx])
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
        # hyperparam ∀:
        const_r_xy = 1.4172946
        V_min = minimum(V)
        V_l = V[argmax(R)]
        Δ = V_l - V_min

        # neumbasis:
        best_scale = 0.
        min_rmse = min_armse = Inf;
        for scale ∈ scaling
            ## train:
            θ, A, q = linratpot_neumbasis(V_train, R_train, const_r_xy, scale)
            ## test:
            _, A, q = linratpot_neumbasis(V_test, R_test, const_r_xy, scale)
            V_pred = A*θ
            rmse = f_RMSE(V_test, V_pred)
            armse = Δ*f_RMSE(δ_dissociate(V_test, V_pred, f_ΔV(V_pred, V_l, V_min)))
            if armse < min_armse
                min_rmse = rmse
                min_armse = armse
                best_scale = scale
            end
        end
        println("neumbasis scale = ",best_scale)
        println("neumbasis RMSE = ",min_rmse)
        println("neumbasis aRMSE = ",min_armse)
        write(io, "neumbasis scale = ",string(best_scale), "\n")
        write(io, "neumbasis RMSE = ",string(min_rmse), "\n")
        write(io, "neumbasis aRMSE = ",string(min_armse), "\n")
        
        # cheb:
        best_d = 0; best_scale = 0.
        min_rmse = min_armse = Inf;
        for scale ∈ scaling
            for d ∈ 1:30
                ## train:
                #θ, A, q = linratpot_cheb(V_train, R_train, const_r_xy, d, 1)
                θ, A_scaled, A, q = linratpot_cheb_diag(V_train, R_train, const_r_xy, d, scale)
                ## test:
                _, A_scaled, A, q = linratpot_cheb_diag(V_test, R_test, const_r_xy, d, scale)
                V_pred = A*θ
                rmse = f_RMSE(V_test, V_pred)
                armse = Δ*f_RMSE(δ_dissociate(V_test, V_pred, f_ΔV(V_pred, V_l, V_min)))
                if armse < min_armse
                    min_rmse = rmse
                    min_armse = armse
                    best_d = d
                    best_scale = scale
                end
            end
        end
        println("Cheb scale = ",best_scale)
        println("Cheb deg = ",best_d)
        println("Cheb RMSE = ", min_rmse)
        println("Cheb aRMSE = ",min_armse)
        write(io, "Cheb scale =",string(best_scale), "\n")
        write(io, "Cheb deg = ",string(best_d), "\n")
        write(io, "Cheb RMSE = ",string(min_rmse), "\n")
        write(io, "Cheb aRMSE = ",string(min_armse), "\n")

        # BUMP:
        best_N = 0; best_scale = 0.
        min_rmse = min_armse = Inf;
        for scale ∈ scaling
            for N ∈ 1:60
                ## train:
                #θ, A, q = linratpot_BUMP(V_train, R_train, const_r_xy, N, 1)
                θ, A_scaled, A, q = linratpot_BUMP_diag(V_train, R_train, const_r_xy, N, scale)
                ## test:
                _, A_scaled, A, q = linratpot_BUMP_diag(V_test, R_test, const_r_xy, N, scale)
                V_pred = A*θ
                rmse = f_RMSE(V_test, V_pred)
                armse = Δ*f_RMSE(δ_dissociate(V_test, V_pred, f_ΔV(V_pred, V_l, V_min)))
                if armse < min_armse
                    min_rmse = rmse
                    min_armse = armse
                    best_N = N
                    best_scale = scale
                end
            end
        end
        println("BUMP scale = ",best_scale)
        println("BUMP deg = ",best_N)
        println("BUMP RMSE = ", min_rmse)
        println("BUMP aRMSE = ",min_armse)
        write(io, "BUMP scale =",string(best_scale), "\n")
        write(io, "BUMP deg = ",string(best_N), "\n")
        write(io, "BUMP RMSE = ",string(min_rmse), "\n")
        write(io, "BUMP aRMSE = ",string(min_armse), "\n")
    end
    close(io);
end

"""
best degs obtained from above
"""
function linear_ratpots_best()
    homedir = "/users/baribowo/Code/Python/pes/"
    # storage:
    #df_rmse = DataFrame(deg=[], neum=[], cheb=[], bump=[])

    data_list = ["H2", "O2", "OH"]
    file_list = ["data/h2/h2_ground_w.txt", "data/diatomic/o2_data.txt", "data/diatomic/oh_data.txt"]
    # best degrees:
    cheb_d = [18, 3, 6]
    BUMP_N = [28, 3, 44]
    for file_idx ∈ 1:length(file_list)
        println("data = ",data_list[file_idx])
        # load data and split 50:50:
        data = readdlm(homedir*file_list[file_idx])
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
        # hyperparam ∀:
        const_r_xy = 1.4172946
        V_min = minimum(V)
        V_l = V[argmax(R)]
        Δ = V_l - V_min

        # cheb
        d = cheb_d[file_idx]
        ## train:
        θ, A, q = linratpot_cheb(V_train, R_train, const_r_xy, d, 1)
        ## test:
        _, A, q = linratpot_cheb(V_test, R_test, const_r_xy, d, 1)
        V_pred = A*θ
        rmse = f_RMSE(V_test, V_pred)
        armse = Δ*f_RMSE(δ_dissociate(V_test, V_pred, f_ΔV(V_pred, V_l, V_min)))
        println("Cheb deg = ", d)
        println("Cheb RMSE = ", rmse)
        println("Cheb aRMSE = ",armse)
        savefig(plot(R_test, V_test-V_pred, markershape = :+, fmt = :png), homedir*"plot/linear_pairpot/cheb_"*data_list[file_idx]*".png")

        # BUMP:
        N = BUMP_N[file_idx]
        ## train:
        θ, A, q = linratpot_BUMP(V_train, R_train, const_r_xy, N, 1)
        ## test:
        _, A, q = linratpot_BUMP(V_test, R_test, const_r_xy, N, 1)
        V_pred = A*θ
        rmse = f_RMSE(V_test, V_pred)
        armse = Δ*f_RMSE(δ_dissociate(V_test, V_pred, f_ΔV(V_pred, V_l, V_min)))
        println("BUMP deg = ", N)
        println("BUMP RMSE = ", rmse)
        println("BUMP aRMSE = ",armse)
        savefig(plot(R_test, V_test-V_pred, markershape = :+, fmt = :png), homedir*"plot/linear_pairpot/BUMP_"*data_list[file_idx]*".png")
    end
end

function multirestart_PES_features()
    # hyperparam:
    n_atom, n_basis, e_pow = (3, 59, 3) #(5, 59, 3)
    const_r_xy, N, max_deg = (1.4172946, 2, 5)
    idxer = atom_indexer(n_atom)
    println("n_atom = ",n_atom)

    # data loader:
    #homedir = "/users/baribowo/Code/Python/pes/" # for work PC only, julia bash isn't available.
    homedir = "" # default
    
    H_data = readdlm(homedir*"data/h3/h3_data.txt") #H_data = readdlm(homedir*"data/h5/h5_data.txt")  # potential data
    X = npzread(homedir*"data/h3/h3_coord.npy") #X = npzread(homedir*"data/h5/h5_coord.npy")  # atomic coordinates
    
    R = H_data[:,1:end-1]; V = H_data[:, end]
    siz = size(R)[1]
    sub_R = R[1:siz,:];
    sub_V = V[1:siz];
    sub_X = X[1:siz, :, :];
    println("data size =",siz)

    # data split:
    #=
    idxes = shuffleobs(1:siz) # shuffle indexes
    id_train, id_test = splitobs(idxes, at=0.8) # split train and test indexes
    writedlm(homedir*"data/h3/index_train_H3.csv", id_train)
    writedlm(homedir*"data/h3/index_test_H3.csv", id_test)
    =#
    # reuse split, usually for re-optimization, comment "data split" block when doing so (no automatic input? hazukashi~ shi~ shi~):
    id_train = vec(readdlm(homedir*"data/h3/crossval_indices_1_train.txt", Int)) #id_train = vec(readdlm(homedir*"data/h5/crossval_indices_1_train.txt", Int)) #id_train = vec(readdlm(homedir*"data/h3/index_train_H3.csv", Int))
    id_test = vec(readdlm(homedir*"data/h3/crossval_indices_1_test.txt", Int)) #id_test = vec(readdlm(homedir*"data/h5/crossval_indices_1_test.txt", Int)) #id_test = vec(readdlm(homedir*"data/h3/index_test_H3.csv", Int))

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
                            R_train, V_train, Θ_vec, show_trace=false, maxIter=200)
            V_pred = f_eval_wrapper_BUMP(res.param, R_train, X_train, idxer, const_r_xy, n_basis, N, e_pow, max_deg)
            # write intermediate params to file 
            writedlm(homedir*"params/h3/multirestart/c_"*string(iter)*".csv", Θ_vec) #writedlm(homedir*"params/h5/multirestart/c_"*string(iter)*".csv", Θ_vec)
            
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
    # save param with best RMSE:
    writedlm(homedir*"params/h3/minimizer_H3_cvindices.csv", Θ_min) #writedlm(homedir*"params/h5/minimizer_H5.csv", Θ_min)
    x = readdlm(homedir*"params/h3/minimizer_H3_cvindices.csv", '\t') #x = readdlm(homedir*"params/h5/minimizer_H5.csv", '\t')
    
    # test data evaluation:
    V_pred = f_eval_wrapper_BUMP(x, R_test, X_test, idxer, const_r_xy, n_basis, N, e_pow, max_deg)
    for i=1:length(V_test)
        println(V_test[i]," ",V_pred[i])
    end
    println("min train RMSE = ",min_rmse)
    println("test RMSE = ", f_RMSE(V_pred, V_test))
    println("elapsed multirestart time = ",t)
end

"""
multirestart with Automatic Differentiation! and Pre-computed features!!
"""
function multirestart_pre_AD()
    #homedir = "/users/baribowo/Code/Python/pes/" # for work PC only, julia bash isn't available.
    homedir = "" # default

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
    println(rmse," ",armse)

    # data loader:    
    H_data = readdlm(homedir*"data/h3/h3_data.txt") #H_data = readdlm(homedir*"data/h5/h5_data.txt")  # potential data
    X = npzread(homedir*"data/h3/h3_coord.npy") #X = npzread(homedir*"data/h5/h5_coord.npy")  # atomic coordinates
    R = H_data[:,1:end-1]; V = H_data[:, end]
    siz = size(R)[1]
    sub_R = R[1:siz,:];
    sub_V = V[1:siz];
    sub_X = X[1:siz, :, :];
    println("data size =",siz)

    # data split:
    #=
    idxes = shuffleobs(1:siz) # shuffle indexes
    id_train, id_test = splitobs(idxes, at=0.8) # split train and test indexes
    writedlm(homedir*"data/h3/index_train_H3.csv", id_train)
    writedlm(homedir*"data/h3/index_test_H3.csv", id_test)
    =#
    # reuse split, usually for re-optimization, comment "data split" block when doing so (no automatic input? hazukashi~ shi~ shi~):
    id_train = vec(readdlm(homedir*"data/h3/crossval_indices_1_train.txt", Int)) #id_train = vec(readdlm(homedir*"data/h5/crossval_indices_1_train.txt", Int)) #id_train = vec(readdlm(homedir*"data/h3/index_train_H3.csv", Int))
    id_test = vec(readdlm(homedir*"data/h3/crossval_indices_1_test.txt", Int)) #id_test = vec(readdlm(homedir*"data/h5/crossval_indices_1_test.txt", Int)) #id_test = vec(readdlm(homedir*"data/h3/index_test_H3.csv", Int))

    # split data by index:
    R_train = sub_R[id_train,:]; V_train = sub_V[id_train];
    R_test = sub_R[id_test,:]; V_test = sub_V[id_test]
    X_train = sub_X[id_train,:,:]; X_test = sub_X[id_test,:,:]
    println("train size = ",length(V_train))
    println("test size = ", length(V_test))

    # hyperparams for feval of train data:
    max_d = 5; 
    n_basis = 59; n_data, n_d = size(R_train); n_atom = 3
    basis_indexes = basis_index_gen(n_basis) # precompute param vector indexes
    idxer = atom_indexer(n_atom)
    println("n_atom = ",n_atom)

    # precompute basis for training!!:
    Φ = f_pot_pre(R_train, X_train, θ, idxer, const_r_xy, d, max_d, n_basis, n_data, n_d)

    #tuning param:
    ub = 1.; lb = -1.
    n_param = n_basis*6
    Θ = rand(n_param).* (ub-lb) .+ lb # tuning parameter
    
    restarts = 100 # number of restarts for the multirestart method
    min_rmse = Inf
    Θ_min = zeros(n_param)
    V_pred = f_energy_wrap(Θ, Φ, n_basis)

    t = @elapsed begin # timer
        for iter=1:restarts
            # precheck nan:
            while any(isnan.(V_pred)) # reset until no nan:
                println("resetting NaNs!!")
                Θ = rand(n_param).* (ub-lb) .+ lb
                res = LsqFit.curve_fit((Φ, Θ) -> f_energy_wrap(Θ, Φ, n_basis), (Φ, Θ) -> df_energy(Θ, Φ, basis_indexes, n_data, n_atom, n_param),
                                Φ, V_train, Θ, show_trace=false, maxIter=2)
                V_pred = f_energy_wrap(res.param, Φ, n_basis)
            end
            # optimize
            Θ = rand(n_param).* (ub-lb) .+ lb
            res = LsqFit.curve_fit((Φ, Θ) -> f_energy_wrap(Θ, Φ, n_basis), (Φ, Θ) -> df_energy(Θ, Φ, basis_indexes, n_data, n_atom, n_param),
                                Φ, V_train, Θ, show_trace=false, maxIter=1000)
            V_pred = f_energy_wrap(res.param, Φ, n_basis)
            # write intermediate params to file 
            writedlm(homedir*"params/h3/multirestart/c_"*string(iter)*".csv", Θ) #writedlm(homedir*"params/h5/multirestart/c_"*string(iter)*".csv", Θ)
            
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
    # save param with best RMSE:
    writedlm(homedir*"params/h3/minimizer_H3_cvindices.csv", Θ_min) #writedlm(homedir*"params/h5/minimizer_H5.csv", Θ_min)
    x = readdlm(homedir*"params/h3/minimizer_H3_cvindices.csv", '\t') #x = readdlm(homedir*"params/h5/minimizer_H5.csv", '\t')
    
    # test data evaluation:
    n_data, n_d = size(R_test)
    Φ = f_pot_pre(R_test, X_test, θ, idxer, const_r_xy, d, max_d, n_basis, n_data, n_d)
    V_pred = f_energy_wrap(x, Φ, n_basis)
    #=
    for i=1:length(V_test)
        println(V_test[i]," ",V_pred[i])
    end
    =#
    println("min train RMSE = ",min_rmse)
    println("test RMSE = ", f_RMSE(V_pred, V_test))
    println("elapsed multirestart time = ",t)
end