
using JLD2

include("get_scenarios.jl")
include("create_scenarios_prod.jl")

"""
---------------------------------------------------------------------------------------------
Parameters - can be modified 
---------------------------------------------------------------------------------------------
"""
nb_w = 500
nb_d = 1
years = 20
H = 6

"""
---------------------------------------------------------------------------------------------
Should not be modified
---------------------------------------------------------------------------------------------
"""

Q = 11
Q_avarie = 5

include("create_substation_A06.jl")
include("simulation multiple comps.jl")

delta_file = "delta_AO0.jld2"
p_file = "p_A06.jld2"

#create_δ(C,h_optim,pr,n,d,H,P_daily,delta_file,P_evac,Q_avarie)
#create_p(C,H,n,P_period,p_file)

@load "delta_AO0.jld2"

@load "p_A06.jld2"

policies_file = "policies_AO6.csv"
#open("policies_AO6.csv", "w") do f
    # ouvrir en mode écriture tronque le fichier
#end


simu_file = "simulation_AO6.csv"
#open("simulation_AO6.csv", "w") do f
    # ouvrir en mode écriture tronque le fichier
#end

cost_file = "cost_AO6.csv"
#open("cost_AO6.csv", "w") do f
    # ouvrir en mode écriture tronque le fichier
#end


# fluid approximation
cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()
collect_policies(policies_file,C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,Q_avarie,P_daily,simu_file,policies_file,"fluid",δ,p,cost_file,false,C)

# do not delete otherwise keep policies in cache
#cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()

#heuristic (benchmark1)
#simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily,simu_file_1,policies_file_1,"benchmark1",δ_w,p,cost_file_1,false,C)

# do not delete otherwise keep policies in cache
#cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()

# optimize without accessibility (benchmark2)
# collect_policies(policies_file_2, C)
# simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily,simu_file_2,policies_file_2,"fluid",δ_no_w,p,cost_file_2,false,C)


#BuildPlot("costs.csv")
