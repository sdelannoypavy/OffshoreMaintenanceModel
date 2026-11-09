
using JLD2

include("get_scenarios.jl")
include("parameters_weakly_coupled.jl")
years = 30
Q = 6
L = 4

include("create_simple_pole.jl")
include("simulation multiple comps.jl")


function create_γ(C,h_optim,pr,n,d,T,P_daily,delta_file,nb_time,P_evac)
# γ remplaces δ

    n_max = maximum(n)

    γ = [0.0 for z in 1:6, m in 1:nb_time, κ in 0:Q, t in 1:T] # z in 1:6 because 6 power levels

    nb_scen = size(h_optim, 2)

    for z in 1:6

        last_state = [1 for c in 1:C]

        if z == 6 #failure
            for c in 1:C
                if d[c] <= 2 #we suppose it is the maintenance with the lowest cost
                #if d[c] <= 2
                    last_state[c] = n[c]
                end
            end
        end

        if 2 <= z <= 5 #intermediate level for fan
            last_state[3] = 2 + z
        end

        for m_global in 1:nb_time
            for κ in 0:Q
                for t in 1:T

                    T_per = 62 - count(==( -1 ), h_optim[t,1,:]) 

                    local Policies_m_t_list = []

                    for c in 1:C

                        if length_m[m_global] >= d[c]
                            m = 1
                        else
                            m = 0
                        end

                        local Policies_m_t = create_schedule_m(m,t_start,T_per)
                        push!(Policies_m_t_list, Policies_m_t)
                    end

                    local Policies_κ_t = create_schedule_κ(κ,t_start,T_per)

                    local cost = 0.0
                    
                    for scen in 1:nb_scen #average on weather scenarios

                        local h_t = h_optim[t,scen,:]
                        local pr_t = pr[t,:]
                        cost_new, _ = simulate_period(Policies_m_t_list,Policies_κ_t,last_state,h_t,pr_t,n,d,P_daily)
                        cost += (1/nb_scen) * cost_new

                    end

                    γ[z,m_global,κ+1,t] = cost

                end
            end
        end

    end

    for i in eachindex(γ)
        if abs(γ[i]) <tol
            γ[i] = 0.0
        end
    end

    @save delta_file γ

end



function PLNE_borne_inf(H,C,x0,k0,t0,γ,p,n)

    Tmax = H

    α = create_α(x0,k0,C)

    model = Model(optimizer)
    #set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, ρ[z=1:6, m = 1:nb_time, κ = 0:Q, t = t0:Tmax] >= 0)    

    @variable(model, q[c = 1:C, x = 1:n[c], k = 0:Q, m = 0:1, κ = 0:Q, t = t0:Tmax] >= 0)     
    @variable(model, A[κ = 0:Q, m in 1:nb_time, t = t0:Tmax] >= 0)   
    @variable(model, μ[m = 1:nb_time, t = t0:Tmax] >= 0)  

    M = 1.0

    @objective(model, Min, M*sum(γ[z,m,κ+1,t]*ρ[z,m,κ,t] for z in 1:6, m in 1:nb_time, κ in 0:Q, t in t0:Tmax))

    # do we really want one vector of probability transitions for each t?
    @constraint(model, [c = 1:C, t = (t0+1):Tmax, x = 1:n[c], k = 0:Q], sum(q[c,x,k,m,κ,t] for m in 0:1, κ in 0:Q) 
        >= sum(p[c,x′,k′+1,x,k+1,m+1,κ+1,t-1]*q[c,x′,k′,m,κ,t-1] for x′ in 1:n[c], k′ in 0:Q, m in 0:1, κ in 0:Q))

    # ensure that if we do a maintenance we also do all shorter maintenances
    is_shorter = zeros(C, 4)
    for c in 1:C
        for l in 1:L
            if d[c] <= length_m[l]
                is_shorter[c,l] = 1
            end
        end
    end

    for c in 1:C
        @constraint(model, [κ = 0:Q, t = t0:Tmax], sum(q[c,x,k,1,κ,t] for x in 1:n[c], k in 0:Q) == sum(is_shorter[c,m]*A[κ,m,t] for m in 1:4))
        @constraint(model, [κ = 0:Q, t = t0:Tmax], sum(q[c,x,k,0,κ,t] for x in 1:n[c], k in 0:Q) == sum((1-is_shorter[c,m])*A[κ,m,t] for m in 1:4))
    end

    @constraint(model, [c = 1:C, x = 1:n[c], k = 0:Q], sum(q[c,x,k,m,κ,t0] for m in 0:1, κ in 0:Q) == α[c,x,k+1])

    @constraint(model, [c = 1:C, x = 1:n[c], k = 0:(Q-1), m = 0:1, κ = (k+1):Q, t = t0:Tmax], q[c,x,k,m,κ,t] == 0)


    # ensuring consistency of state of additionnal component

    # a modifier sur l'autre
    for c in 1:C
        # we pay bause state 6 and maintenance
        @constraint(model, [κ = 0:Q, t = t0:Tmax], sum(q[c,n[c],k,1,κ,t] for x in 1:n[c], k in 0:Q) <= sum(is_shorter[c,l]*ρ[6,l,κ,t] for l in 1:4))
        # we pay bause state 6, no maintenance
        @constraint(model, [κ = 0:Q, t = t0:Tmax], sum(q[c,x,k,0,κ,t] for x in 1:n[c], k in 0:Q) == sum((1-is_shorter[c,l])*A[κ,m,t] for m in 1:4))
        for x in 1:(n[c]-1) # we pay because maintenance, even if not borken
            @constraint(model, [κ = 0:Q, t = t0:Tmax], sum(q[c,x,k,0,κ,t] for x in 1:n[c], k in 0:Q) == sum((1-is_shorter[c,m])*A[κ,m,t] for m in 1:4))
        end
    end

    # we pay because intermediary state of component 3
    @constraint(model, [z = 2:5, m = 0:1, κ = 0:Q, t = t0:Tmax], sum(q[3,2+z,k,m,κ,t] for k in 0:Q) <= sum(ρ[z,l,κ,t] for l in 1:4))

    @constraint(model, [κ = 0:Q, t = t0:Tmax], sum(q[1,n[1],k,1,κ,t] for k in 0:Q) <= ρ[6,4,κ,t] for l in 2:4)
    @constraint(model, [κ = 0:Q, t = t0:Tmax], sum(q[4,n[4],k,1,κ,t] for k in 0:Q) <= sum(ρ[6,l,κ,t] for l in 3:4))


    @constraint(model, [m = 0:1, κ = 0:Q, t = t0:Tmax],  sum(ρ[z,m,κ,t] for z in 1:6) == A[κ,m,t])


    """
    ---------------------------------------------------------------------------------------------
    Optimize
    ---------------------------------------------------------------------------------------------
    """

    optimize!(model)

    status = termination_status(model)
    cost = JuMP.objective_value(model)
#= 
    q_val = [0.0 for c in 1:C, x in 1:maximum(n), k in 0:Q, m in 1:nb_time, κ in 0:Q, t in 1:(Tmax - t0 + 1)]

    for c in 1:C
        for x in 1:n[c], k in 0:Q, m in 1:nb_time, κ in 0:Q, t in t0:Tmax
            q_val[c, x, k+1, m, κ+1, t - t0 + 1] = value(q[c, x, k, m, κ, t])
        end
    end

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
    end     =#

    #return m_opt, κ_opt, cost

    return cost

end

function π2(H,x0,k0,t,h,file_name_policies,δ,p,pr,n,d,P_daily)

    T_per = length(h)

    t0 = (t-1)%6 + 1

    cle = (Tuple(x0), k0, t0)

    if haskey(cache, cle) #already computed
        m_opt, κ_opt = cache[cle]
    else  #not yet computed
        m_opt, κ_opt, _= PLNE_borne_inf(H,C,x0,k0,t0,δ,p,n)
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


k0 = Q
x0 = [1 for c in 1:C]

T = 6*years

#create_p(C,T,n,P_period,"p_30.jld2") #years-6 because of preprocessing in the function 
#create_γ(C,h_optim,pr,n,d,T,P_daily,"delta_30_inf.jld2",nb_time,P_evac)

@load "p_30.jld2"
@load "delta_30_inf.jld2"

cost,q,α = PLNE_borne_inf(12,C,x0,k0,1,γ,p,n)

# check consistency of values of gamma
# the new component takes as many values as there are power levels
# ecrire tests 
# pb sur moment où q est remis à 0? 

# should be rewritten with formulation2? 