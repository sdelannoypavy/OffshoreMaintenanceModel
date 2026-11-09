include("create_simple_pole.jl")
Q = 6
include("simulation multiple comps.jl")

nb_w = 500
nb_d = 1

C = 7


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


P_water99 = build_P(365*MTBF_water*0.99)[:, :, 2] #long theta, do not maintain in state 2
P_water01 = build_P(365*MTBF_water*0.1)[:, :, 2] #short theta, maintain in state 2


P_daily01 = [P_conv,P_water01,P_fan,P_pump,P_trans,P_cool,P_bus]
P_period01 = [P_conv^60,P_water01^60,P_fan^60,P_pump^60,P_trans^60,P_cool^60,P_bus^60]

P_daily99 = [P_conv,P_water99,P_fan,P_pump,P_trans,P_cool,P_bus]
P_period99 = [P_conv^60,P_water99^60,P_fan^60,P_pump^60,P_trans^60,P_cool^60,P_bus^60]

include("simulation multiple comps.jl")

H = 6
Q = 6
#create_δ(C,h,pr,n,d,H,P_daily01,"delta01monopole.jld2",nb_time,P_evac)
create_δ(C,h,pr,n,d,H,P_daily99,"delta99monopole.jld2",nb_time,P_evac)

#create_p(C,H,n,P_period01,"p01monopole.jld2")
create_p(C,H,n,P_period99,"p99monopole.jld2")

"""
@load "delta01monopole.jld2"
δ01 = δ
@load "delta99monopole.jld2"
δ99 = δ

@load "p01monopole.jld2"
p01 = p
@load "p99monopole.jld2"
p99 = p

cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()
collect_policies("policies_sensitivity01.csv",C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily01,"simulation0101.csv","policies_sensitivity01.csv","fluid",δ01,p01,"cost0101.csv",false,C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily99,"simulation0199.csv","policies_sensitivity01.csv","fluid",δ01,p01,"cost0199.csv",false,C)

cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()
collect_policies("policies_sensitivity99.csv",C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily01,"simulation9901.csv","policies_sensitivity99.csv","fluid",δ99,p99,"cost9901.csv",false,C)
simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily99,"simulation9999.csv","policies_sensitivity99.csv","fluid",δ99,p99,"cost9999.csv",false,C) 

#df = CSV.read("cost0101.csv", DataFrame)
#stderr = std(df[:,2]) /sqrt(length(df[:,2]))

"""