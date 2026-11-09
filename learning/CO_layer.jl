C = 7
L = 4
learning_dim = (Q+1)*L + C + 1

function period_cost_vector(x,L,C,T)

    # this only computes the first values of the cost vectors, that are cost associated with scheduling decisions over the period
    # the rest of the cost vector is outputed by the neural network 

    θ_full = []

    L = Int(round(L))

    T_per = 62 - count(==( -1 ), h[T,1,:]) 

    # we don't compute the costs in advance because we would need to compute it for each value of the joint state

    for m in 1:L
        for κ in 0:Q
            
            total_cost = 0
            
            main_dec = [0 for i in 1:C]
            for c in 1:C
                if d[c] <= length_m[m]
                    main_dec[c] = 1
                end
            end

            t_start = 1

            Policies_m_t_list = [create_schedule_m(main_dec[c],t_start,T_per) for c in 1:C]
            Policies_κ_t = create_schedule_κ(κ,t_start,T_per)

            for scen in 1:nb_scen
                cost, _ = simulate_period(Policies_m_t_list,Policies_κ_t,x,h[T,scen,:],pr[T,:],n,d,P_daily)
                total_cost += (1/nb_scen) * cost
            end

            push!(θ_full,total_cost)

        end
    end

    @assert length(θ_full) == learning_dim - C - 1

    return θ_full
end

function full_solution(x,k,m,κ,L,C,Q)

    # has been tested

    # solution contains the values of each δ[l,kappa], then the expected end states for each component, then the maintenance days
    # at the end of the strategic schedule 

    max_len = maximum([m[c]*d[c] for c in 1:C])
    l_dec = findfirst(==(max_len), length_m)

    y = []

    δ =  [0.0 for l in 1:L, kappa in 1:(Q+1)]
    for l in 1:L
        for kappa in 1:(Q+1)
            if (κ+1 == kappa) && (l == l_dec)
                δ[l,kappa] = 1.0
            end
            push!(y, δ[l,kappa])
        end
    end


    avg_x_end = [0.0 for c in 1:C]

    for c in 1:C
        if m[c] == 0
            avg_x_end[c] = sum(s*P_period[c][s,Int(round(x[c]))] for s in 1:n[c])
        else
            avg_x_end[c] = 1
        end
        push!(y,avg_x_end[c])
    end

    k_end = k - κ
    push!(y, k_end)

    @assert length(y) == learning_dim

    return y

end

function CO_layer(θ; x, k, L, Q)


    θx = [θ[L*(Q+1) + c] for c in 1:C]
    θk = θ[L*(Q+1) + C + 1]

    L = Int(round(L))

    c = [θ[(Q+1)*(l-1) + κ] for l in 1:L, κ in 1:(k+1)]
    
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    
    @variable(model, δ[1:L, 0:k], Bin)
    @variable(model, x_end[c=1:C, x=1:n[c]] >= 0)
    @variable(model, m[c=1:C], Bin)
    @variable(model, k_end >= 0)
    @variable(model, leng >= 0)

    @constraint(model, sum(δ[l, κ] for l in 1:L, κ in 0:k) == 1)

    # we define a maximizer as demanded by InferOpt.jl
    @objective(model, Max, sum(c[l,κ+1]*δ[l,κ] for l in 1:L, κ in 0:k) + sum(θx[c]*sum(s*x_end[c,s] for s in 1:n[c]) for c in 1:C) + θk*k_end)

    @constraint(model,k_end == k - sum(κ*sum(δ[l, κ] for l in 1:L) for κ in 0:k))
    @constraint(model,leng == sum(length_m[l]*δ[l, κ] for l in 1:L, κ in 0:k))

    d_max = maximum(d)

    @constraint(model, [c in 1:C], m[c] <= 1 + (1/d_max)*(leng - d[c]))

    @constraint(model, [c in 1:C, s in 1:n[c]], !m[c] --> {x_end[c,s] == P_period[c][s,x[c]]})
    
    @constraint(model, [c in 1:C, s in 2:n[c]], m[c] --> {x_end[c,s] == 0.0})
    @constraint(model, [c in 1:C], m[c] --> {x_end[c,1] == 1.0})


    
    optimize!(model)
    
    y = []

    for l in 1:L
        for kappa in 0:Q
            if kappa <= k
                push!(y, value(δ[l,kappa]))
            else
                push!(y, 0.0)
            end
        end
    end

    for c in 1:C
        avg_x_end = sum(s*value(x_end[c,s]) for s in 1:n[c])
        push!(y,avg_x_end)
    end

    push!(y, value(k_end))

    @assert length(y) == learning_dim
    #dot_val = dot(θ,y)
    #println("computed dot :$dot_val")

    return(y)

end

