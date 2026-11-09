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

file_name_policies = "policies.csv"

"""
---------------------------------------------------------------------------------------------
Construct model
---------------------------------------------------------------------------------------------
"""

function create_α(x0,k0,C)

    α = [0.0 for c in 1:C, x in 1:n_max, k in 0:Q]

    for c in 1:C
        α[c,x0[c],k0+1] = 1
    end

    return α

end

# keep computations in memory

function π(H,x0,k0,t,h,file_name_policies,δ,p,pr,n,d,P_daily)

    T_per = length(h)

    t0 = (t-1)%6 + 1

    cle = (Tuple(x0), k0, t0)

    if haskey(cache, cle) #already computed
        m_opt, κ_opt = cache[cle]
    else  #not yet computed
        m_opt, κ_opt, _= PLNE(H,C,x0,k0,t0,δ,p,n)
        println(m_opt)
        println(κ_opt)
        cache[cle] = (m_opt, κ_opt)

        row = DataFrame(reshape([x0; k0; t0; m_opt; κ_opt], 1, :), :auto) 
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

int_m = [3,1,1,2,1,1,1]

function PLNE(H,C,x0,k0,t0,δ,p,n)

    Tmax = min(H + t0, 180)

    α = create_α(x0,k0,C)

    model = Model(optimizer)
    #set_optimizer_attribute(model, "OutputFlag", 0)


    @variable(model, q[c = 1:C, x = 1:n[c], k = 0:Q, m = 1:(C+1), κ = 0:Q, t = t0:Tmax] >= 0)     
    @variable(model, A[κ = 0:Q, m = 1:(C+1), t = t0:Tmax] >= 0)   
    @variable(model, μ[k = 0:Q, t = t0:Tmax] >= 0)  

    M = 100.0

    @objective(model, Min, M*sum(δ[c,x,m,κ+1,t]*q[c,x,k,m,κ,t] for c in 1:C, x in 1:n[c], k in 0:Q, m in 1:int_m[c], κ in 0:Q, t in t0:Tmax) + sum(δ[2,1,2,κ+1,t]*A[κ,2,t] + δ[4,1,3,κ+1,t]*A[κ,3,t] + δ[1,1,4,κ+1,t]*A[κ,4,t] for κ in 0:Q, t in t0:Tmax))

    # do we really want one vector of probability transitions for each t?
    @constraint(model, [c = 1:C, t = (t0+1):Tmax, x = 1:n[c], k = 0:Q], sum(q[c,x,k,m,κ,t] for m in 1:nb_time, κ in 0:Q) 
        >= sum(p[c,x′,k′+1,x,k+1,m,κ+1,t-1]*q[c,x′,k′,m,κ,t-1] for x′ in 1:n[c], k′ in 0:Q, m in 1:nb_time, κ in 0:Q))

    @constraint(model, [c = 1:C, m = 1:nb_time, κ = 0:Q, t = t0:Tmax], sum(q[c,x,k,m,κ,t] for x in 1:n[c], k in 0:Q) == A[κ,m,t])

    @constraint(model, [c = 1:C, x = 1:n[c], k = 0:Q], sum(q[c,x,k,m,κ,t0] for m in 1:nb_time, κ in 0:Q) == α[c,x,k+1])

    @constraint(model, [c = 1:C, x = 1:n[c], k = 0:(Q-1), m = 1:nb_time, κ = (k+1):Q, t = t0:Tmax], q[c,x,k,m,κ,t] == 0)

    @constraint(model, [c = 1:C, k = 0:Q, t = t0:Tmax], sum(q[c,x,k,m,κ,t] for x in 1:n[c], m in 1:nb_time, κ in 0:Q) == μ[k,t])


    """
    ---------------------------------------------------------------------------------------------
    Optimize
    ---------------------------------------------------------------------------------------------
    """

    optimize!(model)

    status = termination_status(model)
    cost = JuMP.objective_value(model)

    κ_opt = argmax([sum(value(A[κ,m,t0]) for m in 1:nb_time) for κ in 0:Q]) - 1
    m_l = argmax([sum(value(A[κ,m,t0]) for κ in 0:Q) for m in 1:nb_time]) 
    m_opt = [(d[c] <= length_m[m_l]) ? 1 : 0 for c in 1:C]


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



