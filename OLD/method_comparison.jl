include("build_MC_utils.jl")
include("create_simple_pole.jl")
include("optimization_one_comp.jl")
include("get_scenarios.jl")
include("create_scenarios_prod.jl")
#we take pump as an exemple


n_pump = n[4]
n = [n_pump]

d_pump = d[4]
d = [d_pump]

P_daily = [P_pump]
P_period = [P_pump^60]

nb_time = 2
length_m = [0,d_pump]

Q = 1
C = 1

P_evac = [100 100 0;]

years = 30
nb_w = 10000
nb_d = 1
nb_scen_Bell = 80

file_name_simu_1 = "simu_compar_1.csv"
file_name_policies_1 = "policy_compar_1.csv"
file_name_simu_2 = "simu_compar_2.csv"
file_name_policies_2 = "policy_compar_2.csv"
cost_file = "cost_compar.csv"

include("simulation multiple comps.jl")

#create_δ(C,h,pr,n,d,6,P_daily,"delta_compar.jld2",nb_time,P_evac)
#create_p(C,6,n,P_period,"p_compar.jld2")

Vals_all, Policies_m_bellman, Policies_κ_bellman = Bellman(years,Q,nb_scen_Bell,h,n_pump,P_pump,d_pump)
@save "Policies_m_compar.jld2" Policies_m_bellman
@save "Policies_κ_compar.jld2" Policies_κ_bellman


@load "Policies_m_compar.jld2" 
@load "Policies_κ_compar.jld2" 

@load "delta_compar.jld2"
δ_pump = δ

@load "p_compar.jld2"

#simulate(nb_w,nb_d,6,h,pr,n,years,Q,d,P_daily,file_name_simu_2,file_name_policies_2,"fluid",δ_pump,p,cost_file,false,C)
simulate(nb_w,nb_d,6,h,pr,n,years,Q,d,P_daily,file_name_simu_1,file_name_policies_1,"bellman",δ_pump,p,cost_file,false,C)

#df = CSV.read("cost_compar.csv", DataFrame, header=false) 
#grouped_mean = combine(groupby(df, :Column1), :Column2 => mean => :Mean_Column2)
