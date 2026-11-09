include("get_scenarios.jl")
include("create_scenarios_prod.jl")


using JLD2

#Parameters

h_no_w =  ones(Int, 180, 1, 62)
h_optim = h # change for h_no_w to optimize without weather scenarios

t_start = 1
margin = 30

tol = 1e-10

function create_schedule_m(m,t_start,T_per)

    Policies_m_t = [0 for t in 1:T_per]
                    
    if m == 1
        Policies_m_t[t_start] = 1
    end

    return Policies_m_t

end

function create_schedule_κ(κ,t_start,T_per)

    Policies_κ_t = [0 for t in 1:T_per]

    if κ > 0
        for q in 1:κ
            Policies_κ_t[t_start + q - 1] = 1
        end
    end

    return Policies_κ_t

end


"""
---------------------------------------------------------------------------------------------
Transition probabilities
---------------------------------------------------------------------------------------------
"""

function create_p(C,T,n,P_period,p_file)
    Tmax_computeδ = T+6
    p = [0.0 for c in 1:C, x in 1:n_max, k in 0:Q, x′ in 1:n_max, k′ in 0:Q, m in 1:2, κ in 0:Q, t in 1:Tmax_computeδ]

    #transition probabilities for 2 months 


    for c in 1:C
        for t in 1:Tmax_computeδ
            for k in 0:Q
                for x in 1:n[c]
                    for κ in 0:k
                        for m in 1:2
                            if m == 2
                                if (t % 6 == 0) #end of the year
                                    p[c,x,k+1,1,Q + 1,m,κ + 1,t] = 1.0 #maintenance
                                else
                                    p[c,x,k+1,1,k - κ + 1,m,κ + 1,t] = 1.0
                                end
                            else
                                for x′ in 1:n[c] 
                                    if (t % 6 == 0) #end of the year
                                        p[c,x,k+1,x′,Q + 1,m,κ + 1,t] = P_period[c][x′,x] #no maintenance
                                    else
                                        p[c,x,k+1,x′,k - κ + 1,m,κ + 1,t] = P_period[c][x′,x]
                                    end
                                end
                            end
                        end

                    end     
                end
            end   
        end
    end

    tol = 1e-10 #replace very small coefficients by 0 to help the solver

    for i in eachindex(p)
        if abs(p[i]) <tol
            p[i] = 0.0
        end
    end

    @save p_file p

end

"""
---------------------------------------------------------------------------------------------
Costs
---------------------------------------------------------------------------------------------
"""

function average_period_cost(T,Policies_m_t,Policies_κ_t,state,h_t,pr,nb_state,d,P,P_evac)

    # simulate states only over a strategic period for one failure scenario, weather scenario given by h, starting from state

    u = [0 for i in 1:T]
    #ong_m = [0 for i in 1:T]
    q = [0 for i in 1:T]

    q[1] = d*round(Int,Policies_m_t[1])

    last_state = zeros(nb_state)
    last_state[state] = 1.0

    if (q[1]>0)&&(h_t[1]==1)
        u[1] = 1
    end

    costs = []

    if q[1] >= 1
        new_cost = pr[1]*(1-Policies_κ_t[1])
    else 
        new_cost = (1-Policies_κ_t[1])*max(pr[1] - sum(last_state[i]*P_evac[i] for i in 1:nb_state),0)
    end 
    push!(costs, new_cost)

    for t in 1:T-1

        if u[t] == 0
            new_state = P[:,:]*last_state # no maintenance
        else 
            new_state = [1.0; zeros(nb_state - 1)]  # maintenance
        end

        q[t+1] = q[t] - u[t] + d*round(Int,Policies_m_t[t+1])
        if (q[t+1]>0)&&(h_t[t+1]==1)
            u[t+1] = 1
        end

        if q[t+1] >= 1
            new_cost = pr[t]*(1-Policies_κ_t[t+1])
        else 
            new_cost =  (1-Policies_κ_t[t+1])*sum(max(pr[t] - P_evac[i], 0)*new_state[i] for i in 1:nb_state)
        end 

        last_state = new_state
        push!(costs, new_cost)
    end

    return sum(costs)

end 


function create_δ(C,h_optim,pr,n,d,T,P_daily,delta_file,nb_time,P_evac)
    Tmax_computeδ = T+6

    n_max = maximum(n)

    δ = [0.0 for c in 1:C, x in 1:n_max, m in 1:2, κ in 0:Q, t in 1:Tmax_computeδ]

    nb_scen = size(h_optim, 2)

    for c in 1:C
        for x in 1:n[c]
            for m_global in 1:2
                for κ in 0:Q
                    for t in 1:Tmax_computeδ

                        T_per = 62 - count(==( -1 ), h_optim[t,1,:]) 

                        m = m_global - 1

                        local Policies_m_t = create_schedule_m(m,t_start,T_per)
                        local Policies_κ_t = create_schedule_κ(κ,t_start,T_per)

                        local cost = 0.0
                        
                        for scen in 1:nb_scen #average on weather scenarios

                            local h_t = h_optim[t,scen,:]
                            local pr_t = pr[t,:]
                            cost += (1/nb_scen) * average_period_cost(T_per,Policies_m_t,Policies_κ_t,x,h_t,pr_t,n[c],d[c],P_daily[c],P_evac[c,:])

                        end

                        δ[c,x,m_global,κ+1,t] = cost

                    end
                end
            end
        end
    end

    for i in eachindex(δ)
        if abs(δ[i]) <tol
            δ[i] = 0.0
        end
    end

    @save delta_file δ

end