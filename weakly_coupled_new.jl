using Gurobi
using JuMP  
using Distributions
using Random
using StatsBase
using CSV 
using DataFrames


include("parameters_weakly_coupled.jl")

const GRB_ENV = Gurobi.Env()
optimizer=() -> Gurobi.Optimizer(GRB_ENV)

"""
---------------------------------------------------------------------------------------------
Construct model
---------------------------------------------------------------------------------------------
"""

function create_α(x0,k0,q_avarie0,C)

    α = [0.0 for c in 1:C, x in 1:n_max, k in 0:Q, q_avarie in 0:Q_avarie]

    for c in 1:C
        α[c,x0[c],k0+1,q_avarie0+1] = 1
    end

    return α

end

# keep computations in memory

function π(H,x0,k0,q_avarie0,t,h,file_name_policies,δ,p,pr,n,d,P_daily)

    T_per = length(h)

    t0 = (t-1)%6 + 1

    cle = (Tuple(x0), k0, q_avarie0, t0)

    if haskey(cache, cle) #already computed
        m_opt, κ_opt = cache[cle]
    else  #not yet computed
        m_opt, κ_opt, _= PLNE(H,C,x0,k0,q_avarie0,t0,δ,p,n)
        println(m_opt)
        println(κ_opt)
        cache[cle] = (m_opt, κ_opt)

        row = DataFrame(reshape([x0; k0; q_avarie0; t0; m_opt; κ_opt], 1, :), :auto) 
        CSV.write(file_name_policies, row; append=true, writeheader=false) 

    end

    cost_min = 100000

    Policies_m_t_list_min = [[0 for t in 1:T_per] for c in 1:C]
    Policies_κ_t_min = [0 for t in 1:T_per] 

    # we do not have to do what follows if no maintenance

    if sum(m_opt) > 0

        for t_start_it in 1:(60 - margin)
            Policies_m_t_list = [create_schedule_m(m_opt[c],t_start_it,T_per) for c in 1:C]
            Policies_κ_t = create_schedule_κ(κ_opt,t_start_it,T_per)

            cost, _ = simulate_period(Policies_m_t_list,Policies_κ_t,q_avarie0,x0,h,pr,n,d,P_daily)

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



function PLNE(H,C,x0,k0,q_avarie0,t0,δ,p,n)

    Tmax = min(H + t0, 180)

    α = create_α(x0,k0,q_avarie0,C)

    model = Model(optimizer)
    #set_optimizer_attribute(model, "OutputFlag", 0)


    @variable(model, q[c = 1:C, x = 1:n[c], q_avarie = 0:Q_avarie, k = 0:Q, m = 0:1, κ = 0:Q, t = t0:Tmax] >= 0)     
    @variable(model, A[κ = 0:Q, m = 1:(C+1), t = t0:Tmax] >= 0)   

    M = 100.0

    @objective(model, Min, M*sum(δ[c,x,q_avarie+1,m+1,κ+1,t]*q[c,x,q_avarie,k,m,κ,t] for c in 1:C, x in 1:n[c], q_avarie in 0:Q_avarie, k in 0:Q, m in 0:1, κ in 0:Q, t in t0:Tmax))

    # do we really want one vector of probability transitions for each t?
    #@constraint(model, [c = 1:C, t = (t0+1):Tmax, x = 1:n[c], k = 0:Q, q_avarie = 0:Q_avarie], sum(q[c,x,q_avarie,k,m,κ,t] for m in 0:1, κ in 0:Q) 
    #    >= sum(avarie[c,x′,q_avarie′+1,q_avarie+1,m+1,κ+1,t-1]*p[c,x′,k′+1,x,k+1,m+1,κ+1,t-1]*q[c,x′,q_avarie′,k′,m,κ,t-1] for x′ in 1:n[c], q_avarie′ in 0:Q_avarie, k′ in 0:Q, m in 0:1, κ in 0:Q))
    @constraint(model, [c = 1:C, t = (t0+1):Tmax, x = 1:n[c], k = 0:Q, q_avarie = 0:Q_avarie], sum(q[c,x,q_avarie,k,m,κ,t] for m in 0:1, κ in 0:Q) 
        >= sum(p[c,x′,k′+1,x,k+1,m+1,κ+1,t-1]*q[c,x′,q_avarie,k′,m,κ,t-1] for x′ in 1:n[c], k′ in 0:Q, m in 0:1, κ in 0:Q))    
    @constraint(model, [c = 1:C, κ = 0:Q, t = t0:Tmax], sum(q[c,x,q_avarie,k,1,κ,t] for x in 1:n[c], k in 0:Q, q_avarie in 0:Q_avarie) == A[κ,c,t])
    @constraint(model, [c = 1:C, κ = 0:Q, t = t0:Tmax], sum(q[c,x,q_avarie,k,0,κ,t] for x in 1:n[c], k in 0:Q, q_avarie in 0:Q_avarie) == sum(A[κ,i,t] for i in 1:(C+1) if i != c))

    @constraint(model, [c = 1:C, x = 1:n[c], k = 0:Q, q_avarie in 0:Q_avarie], sum(q[c,x,q_avarie,k,m,κ,t0] for m in 0:1, κ in 0:Q) == α[c,x,k+1,q_avarie+1])
    @constraint(model, [c = 1:C, x = 1:n[c], q_avarie = 0:Q_avarie, k = 0:(Q-1), m = 0:1, κ = (k+1):Q, t = t0:Tmax], q[c,x,q_avarie,k,m,κ,t] == 0)

    # should we introduce constraint enforcing sum A = 1 ? 
    
    """
    ---------------------------------------------------------------------------------------------
    Optimize
    ---------------------------------------------------------------------------------------------
    """
    optimize!(model)

    status = termination_status(model)
    cost = JuMP.objective_value(model)

    κ_opt = argmax([sum(value(A[κ,m,t0]) for m in 1:(C+1)) for κ in 0:Q]) - 1
    m_l = argmax([sum(value(A[κ,m,t0]) for κ in 0:Q) for m in 1:(C+1)]) 
    m_opt = [(m_l == c) ? 1 : 0 for c in 1:C]


    if (status == MOI.OPTIMAL) || (status == MOI.LOCALLY_SOLVED)
        println("total cost: ",cost)
        println("optimal κ at step 1: ", κ_opt)
        println("optimal m at step 1: ",  m_opt)
    else
        println("Aucune solution optimale trouvée.")
        println(status)
    end    

    return m_opt, κ_opt, cost

end



