include("build_MC_utils.jl")

#components: converter, water circuit, fans, pumps, transformer, cooling, busbar
d = [6, 1, 1, 2, 1, 1, 1] 

comp_cost = [100.0, 50.0, 20.0, 30.0, 100., 20.0, 100.0]

n_max = maximum(n) #we can create arrays with irregular size

P_evac = [100.0 for c in 1:C, x in 1:n_max]
for c in 1:C
    for x in n[c]:n_max
        P_evac[c,x] = 0.0
    end
end


P_evac[3,1:8] = [100.0, 100.0, 100.0, 80.0, 60.0, 40.0, 20.0, 0.0] #linear decrease for the fan

function Capacity(state_list,n,C,P_evac,q)
    if sum(q[c] for c in 1:C) >= 1 #at least one ongoing maintenance
        return 0
    else 
        return minimum([P_evac[c,state_list[c]] for c in 1:C])
    end
end

P_daily = [P_conv,P_water,P_fan,P_pump,P_trans,P_cool,P_bus]
P_period = [P_conv^60,P_water^60,P_fan^60,P_pump^60,P_trans^60,P_cool^60,P_bus^60]

length_m = sort(unique(d))
insert!(length_m, 1, 0)
nb_time = length(length_m)