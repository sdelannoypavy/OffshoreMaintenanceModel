using JLD2

include("get_scenarios.jl")
include("create_scenarios_prod.jl")
include("shallow_prices.jl")

"""
---------------------------------------------------------------------------------------------
Parameters - can be modified 
---------------------------------------------------------------------------------------------
"""
nb_w = 100
nb_d = 1
years = 20
H = 6

# Shallow price policy parameter
lambda_fn = (m, k) -> k == 5 ? ((m in (3,4)) ? 1 : 5*(m - 1)) : 10000

"""
---------------------------------------------------------------------------------------------
Should not be modified
---------------------------------------------------------------------------------------------
"""

include("create_simple_pole.jl")
include("simulation multiple comps.jl")

Q = 5

d = [1, 1, 1, 1, 1, 1, 1]
nb_time = 2

# create_δ(C,h_optim,pr,n,d,H,P_daily,"delta_shallow.jld2",nb_time,P_evac)

@load "delta_shallow.jld2" δ

@load "p.jld2" p

policies_file = "policies_shallow_price.csv"
open("policies_shallow_price.csv", "w") do f
    # ouvrir en mode écriture tronque le fichier
end


simu_file_fluid = "simulation_shallow_price_fluid.csv"
open("simulation_shallow_price_fluid.csv", "w") do f
    # ouvrir en mode écriture tronque le fichier
end

policies_file_fluid = "policies_shallow_price_fluid.csv"
open("policies_shallow_price_fluid.csv", "w") do f
    # ouvrir en mode écriture tronque le fichier
end


simu_file = "simulation_shallow_price.csv"
open("simulation_shallow_price.csv", "w") do f
    # ouvrir en mode écriture tronque le fichier
end

cost_file = "cost_shallow_price.csv"
open("cost_shallow_price.csv", "w") do f
    # ouvrir en mode écriture tronque le fichier
end

# Compute shallow prices and repair gains
println("Computing shallow prices and repair gains...")
V = compute_all_optimal_costs(δ, p, n)
G = one_step_cont_repair_gain(V, n)
println("Shallow prices computed.")

# Create wrapper policy function for shallow price policy
"""
    π_shallow_price_wrapper(x0, k0, t)

Wrapper for shallow_price_policy that matches the signature expected by simulate function.
Uses the global G dictionary and lambda_threshold parameter.
"""
function π_shallow_price(x0, k0, t)
    
    T_per = length(h)
    
    # Call shallow price policy
    Policies_m_t_list, Policies_κ_t = shallow_price_policy_fixed_lambda(
        x0, k0, t, G, n, C, d, δ, lambda_threshold
    )
    
    
    return Policies_m_t_list, Policies_κ_t
end

# Shallow price approximation
cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()
collect_policies(policies_file, C)
simulate(nb_w, nb_d, H, h, pr, n, years, Q, d, P_daily, simu_file, policies_file, 
         "shallow_price", δ, p, cost_file, false, C)

# do not delete otherwise keep policies in cache
cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()

#fluid
#simulate(nb_w, nb_d, H, h, pr, n, years, Q, d, P_daily, simu_file_fluid, policies_file_fluid, 
#         "fluid", δ, p, cost_file, false, C)



#BuildPlot("costs.csv")




# optimize lambda
