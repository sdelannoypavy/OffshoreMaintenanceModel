# we use the second version of the fluid approximation
C = 7
Q = 6
L = 4

include("create_simple_pole.jl")
include("simulation multiple comps.jl")
include("borne_inf.jl")

nb_w = 500
nb_d = 1


nb_state = 3
P_evac = vcat([100.0 for i in 1:2], [0.0]) # vector of capacity levels

years = 30

MTBF_water = 43

function build_P(theta)
    # theta = mean time in state 2

    P = zeros(3, 3, 2)

    for i in 1:3
        P[1,i,1] = 1.0 
    end

    mean_time_state_1 = 365*MTBF_water - theta 
        
    P[1,1,2] = 1.0 - 1/mean_time_state_1
    P[2,1,2] = 1/mean_time_state_1
        
    P[2,2,2] = 1.0 - 1/theta
    P[3,2,2] = 1/theta

    P[3,3,2] = 1.0

    return P
end



#components: converter, water circuit, fans, pumps, transformer, cooling, busbar
C = 7
n = [12, 3, 8, 3, 3, 3, 3]
d = [6, 1, 1, 2, 1, 1, 1] 

n_max = maximum(n)
T = 6


P_evac = [100.0 for c in 1:C, x in 1:n_max]
for c in 1:C
    for x in n[c]:n_max
        P_evac[c,x] = 0.0
    end
end


P_evac[3,1:8] = [100.0, 100.0, 100.0, 80.0, 60.0, 40.0, 20.0, 0.0] #linear decrease for the fan

function Capacity(state_list,n,C,P_evac,q)
    
    if sum(q[c] for c in 1:C) >= 1 #at least one ongoing maintenance
        return  0
    else 
        return minimum([P_evac[c,state_list[c]] for c in 1:C])
    end
end

theta_true = (1/p_pump)/(43*365)

P_water99 = build_P(365*MTBF_water*0.99)[:, :, 2] #long theta, do not maintain in state 2
P_water01 = build_P(365*MTBF_water*0.1)[:, :, 2] #short theta, maintain in state 2
P_water67 = build_P(365*MTBF_water*theta_true)[:, :, 2] 

P_daily01 = [P_conv,P_water01,P_fan,P_pump,P_trans,P_cool,P_bus]
P_period01 = [P_conv^60,P_water01^60,P_fan^60,P_pump^60,P_trans^60,P_cool^60,P_bus^60]

P_daily67 = [P_conv,P_water67,P_fan,P_pump,P_trans,P_cool,P_bus]
P_period67 = [P_conv^60,P_water67^60,P_fan^60,P_pump^60,P_trans^60,P_cool^60,P_bus^60]

P_daily99 = [P_conv,P_water99,P_fan,P_pump,P_trans,P_cool,P_bus]
P_period99 = [P_conv^60,P_water99^60,P_fan^60,P_pump^60,P_trans^60,P_cool^60,P_bus^60]

include("simulation multiple comps.jl")

H = 6
Q = 6
#create_δ(C,h,pr,n,d,H,P_daily01,"delta01monopole.jld2",nb_time,P_evac)
#create_δ(C,h,pr,n,d,H,P_daily99,"delta99monopole.jld2",nb_time,P_evac)
#create_δ(C,h,pr,n,d,H,P_daily67,"delta67monopole.jld2",nb_time,P_evac)

#create_γ(C,h_optim,pr,n,d,T,P_daily01,"gamma01monopole.jld2",nb_time,P_evac)
#create_γ(C,h_optim,pr,n,d,T,P_daily99,"gamma99monopole.jld2",nb_time,P_evac)
#create_γ(C,h_optim,pr,n,d,T,P_daily67,"gamma67monopole.jld2",nb_time,P_evac)

create_p(C,H,n,P_period01,"p01monopole.jld2")
create_p(C,H,n,P_period99,"p99monopole.jld2")
create_p(C,H,n,P_period67,"p67monopole.jld2")

@load "gamma01monopole.jld2"
δ01 = γ
@load "gamma99monopole.jld2"
δ99 = γ
@load "gamma67monopole.jld2"
δ67 = γ

@load "p01monopole.jld2"
p01 = p
@load "p99monopole.jld2"
p99 = p
@load "p67monopole.jld2"
p67 = p

cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()
collect_policies("policies_sensitivity267.csv",C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily01,"simulation26701.csv","policies_sensitivity267.csv","fluid2",δ67,p67,"cost26701.csv",false,C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily99,"simulation26799.csv","policies_sensitivity267.csv","fluid2",δ67,p67,"cost26799.csv",false,C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily67,"simulation26767.csv","policies_sensitivity267.csv","fluid2",δ67,p67,"cost26767.csv",false,C)

cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()
collect_policies("policies_sensitivity201.csv",C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily01,"simulation20101.csv","policies_sensitivity201.csv","fluid2",δ01,p01,"cost20101.csv",false,C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily99,"simulation20199.csv","policies_sensitivity201.csv","fluid2",δ01,p01,"cost20199.csv",false,C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily67,"simulation20167.csv","policies_sensitivity201.csv","fluid2",δ01,p01,"cost20167.csv",false,C)

cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()
collect_policies("policies_sensitivity299.csv",C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily01,"simulation29901.csv","policies_sensitivity299.csv","fluid2",δ99,p99,"cost29901.csv",false,C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily99,"simulation29999.csv","policies_sensitivity299.csv","fluid2",δ99,p99,"cost29999.csv",false,C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily67,"simulation29967.csv","policies_sensitivity299.csv","fluid2",δ99,p99,"cost29967.csv",false,C)

#df = CSV.read("cost0101.csv", DataFrame)
#stderr = std(df[:,2]) /sqrt(length(df[:,2]))

#retry first fluid approximation since p was wrong 
#verifier pourquoi on ne fait pas toutes les maintenances en fin d'année 