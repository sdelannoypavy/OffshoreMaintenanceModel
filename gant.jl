using Plots
using DataFrames
using ColorSchemes
using Statistics
using DSP 

include("get_scenarios.jl")
include("create_scenarios_prod.jl")
include("create_simple_pole.jl")
include("simulation multiple comps.jl")
include("create_scenarios.jl")
@load "delta.jld2"

@load "p.jld2"

legendfont = font(15)

# =======================
# Exemple de données
# =======================

# Suppose qu'on ait 7 composants et 100 jours
n_components = 7
nb_per = 14
n_days = nb_per*62
components = 1:n_components
times = 1:n_days
start_state = [3, 1, 1, 3, 1, 1, 1] # état initial des composants
bar_list = []

h_plot = [daily_means.daily_mean[t] for t in times]  
h_bin = [[] for s in 1:80]          
cost = [0.0 for t in 1:n_days] 

# Exemple : matrice de dégradation ∈ [0,1] (0 = neuf, 1 = très dégradé)
degradation = [0.0 for i in 1:n_components, t in 1:n_days]
# Exemple : jours de maintenance gratuite et payante
# (1 = maintenance effectuée ce jour-là)
maintenance_free = falses(n_components, n_days)
maintenance_paid = falses(n_components, n_days)

function simulate_period(Policies_m_t_list,Policies_κ_t,last_state,h,pr,n,d,P_daily,t_curr)

        # simulate states only over a strategic period for one failure scenario, weather scenario given by h, starting from state

        T_per = 62 - count(==( -1 ), h)

        for c in 1:C
            degradation[c,t_curr] = (last_state[c] - 1)/(n[c]-1)
        end

        u = [0 for c in 1:C, i in 1:T_per]
        q = [0 for c in 1:C, i in 1:T_per]


        for c in 1:C
            if (q[c,1]>0)&&(h[1]==1)
                u[c,1] = 1
            end
        end
    
        costs = []
        new_cost = (1 - Policies_κ_t[1]) * max(pr[1] - Capacity(last_state,n,C,P_evac,q[:,1]), 0)
        cost[t_curr] = new_cost

        push!(costs, new_cost)

        for c in 1:C
            q[c,1] = d[c]*round(Int,Policies_m_t_list[c][1])
            if u[c,1] == 1  
                if Policies_κ_t[1] == 1
                    maintenance_free[c, tcour] = true
                else
                    maintenance_paid[c, tcour] = true
                end
            end
        end


    
        for t in 1:T_per-1

            new_state = [1 for i in 1:C]

            t_curr += 1

    
            for c in 1:C
                if u[c,t] == 0 #no maintenance
                    probas = P_daily[c][:, last_state[c]]  
                    new_state[c] = sample(1:n[c], Weights(probas))     
                else       
                    new_state[c] = 1
                end
            end
        
    
            for c in 1:C
                q[c,t+1] = q[c,t] - u[c,t] + d[c]*round(Int,Policies_m_t_list[c][t+1])
                if (q[c,t+1]>0)&&(h[t+1]==1)
                    u[c,t+1] = 1
                end
            end

            new_cost = (1 - Policies_κ_t[t+1]) * max(pr[t+1] - Capacity(new_state,n,C,P_evac,q[:,t+1]),0)
    
            last_state = new_state
            push!(costs, new_cost)
            cost[t_curr] = new_cost

            for c in 1:C
                degradation[c,t_curr] = (last_state[c] - 1)/(n[c]-1)
                if u[c,t+1] == 1  
                    if Policies_κ_t[t+1] == 1
                        maintenance_free[c, t_curr] = true
                    else
                        maintenance_paid[c, t_curr] = true
                    end
                end
            end

        end
    
        for c in 1:C
            if u[c,T_per] == 0 #no maintenance
                probas = P_daily[c][:, last_state[c]]   
                last_state[c] = sample(1:n[c], Weights(probas))     
            else       
                last_state[c] = 1
            end
        end

        

        t_curr += 1

    return sum(costs), last_state, t_curr

end 

cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()

# optimize without accessibility (benchmark2)
collect_policies("policies.csv",C)

function simulate_concession_period(H,h, pr, n, nb_per, Q, d, P_daily,file_name_policies,δ,p,C, start_state)

    # simulate stationary policies

    # simulate states over the entire concession period

    total_cost = 0

    last_state = start_state#at the beginning of the concession period, all components are new
    k = Q #we have not use any free maintenance day yet

    t_curr = 1


    for t in 1:nb_per
        
        Policies_m_t_list, Policies_κ_t = π(H, last_state, k, t, h[t,1,:], file_name_policies, δ, p, pr, n, d, P_daily)

        m_opti = [sum(Policies_m_t_list[c]) for c in 1:C]
        κ_opti = sum(Policies_κ_t) 

        for s in 1:80
            for t_day in 1:62
                push!(h_bin[s],max(h[t,s,t_day],0))
            end
        end

        new_cost, new_state, t_curr = simulate_period(Policies_m_t_list,Policies_κ_t,last_state,h[t,1,:],pr[t,:],n,d,P_daily, t_curr)

        if t%6 == 0
            k = Q #end of the year
            push!(bar_list, t_curr)
        else
            k -= κ_opti
        end
 
        last_state = new_state

    end

end


H = 6
Q = 6
simulate_concession_period(H,h, pr, n, nb_per, Q, d, P_daily,file_name_policies,δ,p,C,start_state)


# =======================
# Tracé
# =======================

# Créer une heatmap pour la dégradation
p1 = heatmap(
    times,
    components,
    degradation;
    color=cgrad([:white, :red]),
    xlabel="Time (days)",
    ylabel="Component",
    colorbar_title="Degradation level",
    colorbar_ticks=false,
    colorbar_orientation=:horizontal,
    legend =false,
    yflip=true,  # pour que le composant 1 soit en haut
    guidefont=font(15),
    legendfont = font(15),
)

# Ajouter les maintenances (blocs colorés)
for i in 1:n_components
    for t in times
        if maintenance_free[i, t]
            # Bloc vert : maintenance gratuite
            plot!(p1, [t, t], [i-0.2, i+0.2], lw=6, color=:green, label=false)
        elseif maintenance_paid[i, t]
            # Bloc orange : maintenance payante
            plot!(p1, [t, t], [i-0.2, i+0.2], lw=6, color=:orange, label=false)
        end
    end
end

vline!(p1, bar_list; color=:black, lw=1.5, ls=:dash, label="New year", legend=true)
# Ajout d'entrées de légende pour les types de maintenance
plot!(p1, [NaN], [NaN], color=:green, lw=6, label="Free maintenance", legend=:right)
plot!(p1, [NaN], [NaN], color=:orange, lw=6, label="Paid maintenance")
# =======================
# 2️⃣ Middle plot: wave height
# =======================

p2 = plot(
    times,
    h_plot;
    color=:blue,
    lw=2,
    xlabel="Time (days)",
    ylabel="Wave height (m)",
    guidefont=font(15),
    legendfont = font(15),
    label=false
)
vline!(p2, bar_list; color=:black, lw=1, ls=:dash, label=false)

hline!(p2, [1], color=:red, lw=2, label="Accessibility threshold")

h_sum = [sum(h[,s,t]) for s in 1:length(h_bin)]


#= p4 = plot(
    times,
    h_sum;
    color=:black,
    lw=2,
    xlabel="Time (days)",
    ylabel="accessibility %",
    guidefont=font(15),
    legendfont = font(15),
    label=false,
    yticks=0:50:100
)

vline!(p4, bar_list; color=:black, lw=1, ls=:dash, label=false)
 =#

# =======================
# 3️⃣ Lower plot: cost
# =======================

p5 = plot(
    times,
    cost;
    color=:purple,
    lw=2,
    xlabel="Time (days)",
    ylabel="Cost",
    legend=false,
    guidefont=font(15),
    legendfont = font(15),
    yticks=0:50:100,
)
vline!(p5, bar_list; color=:black, lw=1, ls=:dash, label=false)


# =======================
# Combine all plots
# =======================

layout = @layout [a{0.45h}; b{0.15h}; c{0.05h}]   # 3 rows stacked vertically
plt = plot(
    p1, p2, p5;
    layout=layout,
    size=(1000, 1000),
)

# =======================
# Save and display
# =======================
savefig(plt, "maintenance_schedule.png")
display(plt)