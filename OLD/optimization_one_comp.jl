using Gurobi
using JuMP  
using Distributions
using Random
using StatsBase
#import MathOptInterface as MOI


# compute value function
function V(T,s_init,q_i,S,h,nb_state,P,Q,Vals,d,P_evac)

    println(s_init,q_i)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    #set_optimizer_attribute(model, "MIPFocus", 2)
    #set_optimizer_attribute(model, "Presolve", 2)

    x_0 = zeros(nb_state)      
    x_0[s_init] = 1 

    tol = 1e-1

    @variable(model, 0 - tol <= x[1:S,1:(T+1),1:nb_state] <= 1 + tol) # we need the value at T+1 to get next strategic period value function

    @variable(model, c[1:(T+1),1:S] >= -1000*tol) # c[T+1] is the value function of the next strategic period

    @variable(model, m[1:T], Bin)
    @variable(model, u[1:S,1:T], Bin)
    @variable(model, ong_m[1:S,1:T], Bin) #ongoing maintenance
    @variable(model, q[1:S,1:T] >=0, Int)
    @variable(model, k[1:T], Bin)
    @variable(model, q_f, Int)
    @variable(model, δ[0:Q], Bin) # value of quota as the end of the period formulated using Bin. Usefull to write final cost. 

    alpha = 1.0

    @objective(model, Min, (alpha/S)*sum(c[t,s] for t in 1:(T+1), s in 1:S))

    @constraint(model, [s = 1:S, t = 1:(T+1)], sum(x[s,t,i] for i in 1:nb_state) <= 1 + tol)
    @constraint(model, [s = 1:S, t = 1:(T+1)], sum(x[s,t,i] for i in 1:nb_state) >= 1 - tol)

    @constraint(model, [s = 1:S, t = 1:T], q[s,t] <= d*T)

    margin = 30 #margin to avoid maintenance that we can't finish
    @constraint(model, [t in 1:margin], m[T+1-t] == 0) 

    # state dynamics

    @constraint(model, [s = 1:S, t = 2:(T+1), i = 1:nb_state], !ong_m[s,t-1] --> {x[s,t,i] == sum(P[i,j]*x[s,t-1,j] for j in 1:nb_state)}) # no maintenance
    @constraint(model, [s = 1:S, t = 2:(T+1), i = 2:nb_state], ong_m[s,t-1] --> {x[s,t,i] == 0.0}) #maintenance
    @constraint(model, [s = 1:S, t = 2:(T+1)], ong_m[s,t-1] --> {x[s,t,1] == 1.0}) #maintenance

    @constraint(model, [s = 1:S, i = 1:nb_state], x[s,1,i] == x_0[i])

    # deterministic rule: "maintain as soon as possible"

    @constraint(model, [s = 1:S, i = 1:nb_state], q[s,1] == d*m[1])
    @constraint(model, [s in 1:S, t in 1:(T-1)], q[s,t+1] == q[s,t] - u[s,t] + d*m[t+1])

    @constraint(model, [s = 1:S, t = 1:T], ong_m[s,t] --> {u[s,t] == h[s,t]})
    @constraint(model, [s = 1:S, t = 1:T], !ong_m[s,t] --> {u[s,t] == 0})

    @constraint(model, [s = 1:S, t = 1:T], ong_m[s,t] >= (1/(d*T))*q[s,t])
    @constraint(model, [s = 1:S, t = 1:T], ong_m[s,t] <= q[s,t])


    # cost

    @constraint(model, [s = 1:S, t = 1:T], !ong_m[s,t] --> {c[t,s] == sum(x[s,t,state]*(100 - P_evac[state]) for state in 1:nb_state)})
    @constraint(model, [s = 1:S, t = 1:T], ong_m[s,t] --> {c[t,s] == (1-k[t])*100})


    @constraint(model, q_f == q_i - sum(k[t] for t in 1:T)) 
    @constraint(model, q_f >= 0)

    # ensure delta as the right value
    @constraint(model, sum(δ[j] for j in 0:Q) == 1)
    @constraint(model, q_f == sum(j * δ[j] for j in 0:Q))
    
    @constraint(model, [j in 0:Q, s in 1:S], δ[j] --> {c[T+1,s] == sum(x[s,T+1, i] * Vals[i, j+1] for i in 1:nb_state)})

    
    try
        optimize!(model)

        status = termination_status(model)
        # println("Statut de l'optimisation: $status")

        cost = objective_value(model)


        m = [value(m[t]) for t in 1:T]
        k = [value(k[t]) for t in 1:T]
        
        if (status == MOI.OPTIMAL) || (status == MOI.LOCALLY_SOLVED)
            return(cost, m, k)
        else
            println("Aucune solution optimale trouvée.")
            println(status)
            return(0,[0 for t in 1:T],[0 for t in 1:T])
        end    
    catch e
        println("Erreur pendant la résolution : ", e)
        return(0,[0 for t in 1:T],[0 for t in 1:T])
    end


end


function Bellman(years,Q,S,h,nb_state,P,d)

    Tmax = 6*years

    if S > 0
        h_reduced = reduce_array(h,S)#randomly select S scenarios
    else
        h_reduced = h
    end

    Vals_old = zeros(nb_state, Q+1)

    # Remember all Bellman values
    Vals_all = Array{Float64, 3}(undef, nb_state, Q+1, Tmax)

    # Remember all optimal decisions
    Policies_m = Array{Float64}(undef, nb_state, Q+1, Tmax, 62)
    Policies_k = Array{Float64}(undef, nb_state, Q+1, Tmax, 62)

    for t in 1:Tmax
        Vals_new = zeros(nb_state, Q+1)
        for s in 1:nb_state
            for q in 0:Q
                h_t = h_reduced[Tmax-t+1, :, :]  # taille (S, T)
                T = 62 - count(==( -1 ), h_t[1,:]) # - 1 means ends of the month, so that we can represent months with variable lengths with vectors of the same dimensions
                println(t)

                if (Tmax - t + 1 % 6 == 0) #end year
                    for q in 1:Q
                        for state in 1:nb_sttae
                            Vals_old[state,q] = Vals_old[state,Q+1] #quotas renewed
                        end
                    end
                end

                if S == 0
                    h = ones(1, T)
                    val, m_opt, k_opt = V(T,s,q,1,h,nb_state,P,Q,Vals_old,d,P_evac)
                else
                    val, m_opt, k_opt = V(T,s,q,S,h_t,nb_state,P,Q,Vals_old,d,P_evac)
                end
                Vals_new[s, q+1] = val
                Policies_m[s,q+1,Tmax - t + 1,1:T] = m_opt
                Policies_k[s,q+1,Tmax - t + 1,1:T] = k_opt
            end
        end
        Vals_all[:, :, Tmax - t + 1] = Vals_new 
        Vals_old = Vals_new
    end

    return Vals_all, Policies_m, Policies_k

end

function simulate_period(Policies_m_t,Policies_k_t,state,h,nb_state,d,P)

    # simulate states only over a strategic period for one failure scenario, weather scenario given by h, starting from state

    T = 62 - count(==( -1 ), h)

    u = [0 for i in 1:T]
    ong_m = [0 for i in 1:T]
    q = [0 for i in 1:T]

    q[1] = d*round(Int,Policies_m_t[1])

    last_state = zeros(nb_state)
    last_state[state] = 1.0

    if (q[1]>0)&&(h[1]==1)
        u[1] = 1
    end

    costs = []

    if q[1] >= 1
        new_cost = 100*(1-Policies_k_t[1])
    else 
        new_cost = 100 - sum(last_state[i]*P_evac[i] for i in 1:nb_state)
    end 
    push!(costs, new_cost)

    for t in 1:T-1

        if u[t] == 0
            new_state = P[:,:,2]*last_state # no maintenance
        else 
            new_state = P[:,:,1]*last_state # maintenance
        end

        q[t+1] = q[t] - u[t] + d*round(Int,Policies_m_t[t+1])
        if (q[t+1]>0)&&(h[t+1]==1)
            u[t+1] = 1
        end

        if q[t+1] >= 1
            new_cost = 100*(1-Policies_k_t[t+1])
        else 
            new_cost = 100 - sum(new_state[i]*P_evac[i] for i in 1:nb_state)
        end 

        last_state = new_state
        push!(costs, new_cost)
    end

    if u[T] == 0
        last_state = P[:,:,2]*last_state # no maintenance
    else 
        last_state = P[:,:,1]*last_state # maintenance
    end

    return sum(costs), last_state

end 


function simulate_concession_period(Policies_m, Policies_k, h, nb_state, years, Q, d, P)

    # simulate states over the entire concession period

    total_cost = 0

    nb_product_states = nb_state*(Q+1) #index nb_state*q + state
    last_product_state = zeros(nb_product_states)
    last_product_state[nb_state*Q+1] = 1.0# at the beginning of the concession period the substation is new : state = 1, q = Q

    for t in 1:(6*years)

        new_product_state = zeros(nb_product_states)
        new_cost = 0

        for product_state in 1:nb_product_states 

            q, r = divrem(product_state - 1, nb_state)
            state = r + 1

            Policies_k_t = Policies_k[state,round(Int,q) + 1,t,:]
            sum_k = sum(Policies_k_t)

            Policies_m_t = Policies_m[state,round(Int,q) + 1,t,:]
            h_t = h[t, :]

            cost_state, new_x = simulate_period(Policies_m_t, Policies_k_t, state, h_t, nb_state, d, P)

            q -= sum_k

            for s in 1:nb_state
                new_product_state[round(Int,nb_state*q + s)] += last_product_state[product_state]*new_x[s]
            end

            new_cost += last_product_state[product_state]*cost_state

        end

        #@assert(sum(new_product_state) <= 1 + 10e-3)
        #@assert(sum(new_product_state) >= 1 - 10e-3)
        
        total_cost += new_cost
        last_product_state = new_product_state

    end

    return total_cost

end



function simulate(Policies_m,Policies_k,h,nb_state, years,Q, d, P)

    list_cost = []

    nb_scen = size(h, 2)

    for scen in 1:nb_scen 
        new_cost = simulate_concession_period(Policies_m, Policies_k, h[:,scen,:], nb_state, years, Q, d, P)
        push!(list_cost, new_cost)
    end

    return mean(list_cost)

end


function reduce_array(A::Array{T,3}, l::Int) where T
    n, m, k = size(A)
    @assert l ≤ m "l doit être inférieur ou égal à m"

    # Tirer au hasard l indices uniques parmi 1:m
    selected_cols = sort(sample(1:m, l; replace=false))

    # Extraire les colonnes sélectionnées
    A_reduced = A[:, selected_cols, :]

    return A_reduced
end



#Vals_all, Policies_m, Policies_k = Bellman(years,Q,S,h,nb_state,P)
#state_list = simulate(Policies_m,Policies_k,h,nb_state, years,Q)

#@save "Vals_converter_120725.jld2" Vals_all
#@load "Vals_converter_120725.jld2" 
