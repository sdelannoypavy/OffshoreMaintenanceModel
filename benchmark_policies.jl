# greedy policy
include("CO_layer.jl")

function π_benchmark1(x0,k0,t)
    m_opt = [0 for c in 1:C]
    κ_opt = 0

    if t%6 == 4 #summer

        for c in 1:C 
            if (x0[c] > 1) && (d[c] <= k0)
                m_opt[c] = 1
                κ_opt = k0
            end 
        end
    end

    for c in 1:C
        if x0[c] == n[c] #failure 
            m_opt[c] = 1
            κ_opt = k0
        end
    end

    Policies_m_t_list = [create_schedule_m(m_opt[c],t_start,62) for c in 1:C]
    Policies_κ_t = create_schedule_κ(κ_opt,t_start,62)

    return Policies_m_t_list, Policies_κ_t

end 


#function π_bellman(x0,k0,t)

#    Policies_m_t_list = [Policies_m_bellman[Int(round(x0[1])),Int(round(k0+1)),t,:]]
#    Policies_κ_t = Policies_κ_bellman[Int(round(x0[1])),Int(round(k0+1)),t,:]

#    return Policies_m_t_list, Policies_κ_t

#end 

function π_learning(x0,k0,t,h,file_name_policies,δ,p,pr,n,d,P_daily)

    T_per = length(h)

    t0 = (t-1)%6 + 1
    T = t0
    x = x0
    k = k0

    output = model([x0; k0; t0])
    θ_start = period_cost_vector(x,L,C,T)
    θ = [-θ_start;-K_mult*output]

    y = CO_layer(θ; x, k, L, Q)

    κ_opt = sum([(κ-1)*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
    κ_opt = Int(round(κ_opt))

    l_opt = sum([length_m[l]*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
    m_opt = [0.0 for i in 1:C]
    for c in 1:C
        if d[c] <= l_opt
            m_opt[c] = 1.0
        end
    end

    cle = (Tuple(x0), k0, t0)

    if haskey(cache, cle) #already computed
        m_opt, κ_opt = cache[cle]
    else  #not yet computed
        κ_opt = sum([(κ-1)*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
        κ_opt = Int(round(κ_opt))

        l_opt = sum([length_m[l]*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
        m_opt = [0.0 for i in 1:C]
        for c in 1:C
            if d[c] <= l_opt
                m_opt[c] = 1.0
            end
        end

        cache[cle] = (m_opt, κ_opt)

        row = DataFrame(reshape([x0; k0; t0; m_opt; κ_opt], 1, :), :auto) 
        CSV.write(file_name_policies, row; append=true, writeheader=false)
    end

    cost_min = 100000

    Policies_m_t_list_min = [[0 for t in 1:T_per] for c in 1:C]
    Policies_κ_t_min = [0 for t in 1:T_per] 

    # we do not have to do what follows if no maintenance

    κ_opt = Int(round(κ_opt))

    if sum(m_opt) > 0

        for t_start_it in 1:(60 - margin)
            Policies_m_t_list = [create_schedule_m(m_opt[c],t_start_it,T_per) for c in 1:C]
            Policies_κ_t = create_schedule_κ(κ_opt,t_start_it,T_per)

            cost, _ = simulate_period(Policies_m_t_list,Policies_κ_t,x0,h,pr,n,d,P_daily)

            if cost < cost_min
                cost_min = cost
                Policies_m_t_list_min = Policies_m_t_list
                Policies_κ_t_min = Policies_κ_t
            end

        end

        return Policies_m_t_list_min, Policies_κ_t_min

    else #no maintenance so no need to optimize t_start
        t_start = 1
        Policies_m_t_list = [create_schedule_m(m_opt[c],t_start,T_per) for c in 1:C]
        Policies_κ_t = create_schedule_κ(κ_opt,t_start,T_per)

        return Policies_m_t_list_min, Policies_κ_t_min
    end

end