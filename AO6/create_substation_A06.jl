C = 9
n_bloc = 3

#transformateur de puissance, disjoncteur, barre *3
d = [1 for c in 1:C] 
n = [3 for c in 1:C] # number of blocs still working 

MTBF_tp = 2000
MTBF_dis = 20

p_tp = 1/(365*MTBF_tp) # we have the MTBF for one transformer 
p_dis = 1/(365*MTBF_dis)

function build_P_linear_daily(nb_state,MTBF)

    p = ((nb_state - 1)/MTBF)/365

    P = [1 - p    0       0;
               p        1 - p   0;
               0        p       1]

end

P_tp = build_P_linear_daily(3,MTBF_tp)
P_dis = build_P_linear_daily(3,MTBF_dis)


P_barre = [1 - 0.5*3*p_dis*(1- p_dis)^2  0                                0                
        0.5*3*p_dis*(1- p_dis)^2         1 - 0.5*3*p_dis*(1- p_dis)^2     0                
        0                                0.5*3*p_dis*(1- p_dis)^2         1]

comp_cost = [1.86*100, 0.1*100, 3*0.1*100, 1.86*100, 0.1*100, 3*0.1*100, 1.86*100, 0.1*100, 3*0.1*100] #cost of maintenance for each component
# failure for comp = busbar = failure of the three disjoncteurs, we never replace the busbar itself 

n_max = maximum(n) #we can create arrays with irregular size

P_evac = [100.0 for c in 1:C, x in 1:n_max]
for c in 1:C
    P_evac[c,1:3] = [100.0, 100.0, 0.0]
end

#linear decrease for the fan

function Capacity(state_list,n,C,P_evac,q)
    if sum(q[c] for c in 1:C) >= 1 #at least one ongoing maintenance
        return [0 for bloc in 1:n_bloc]
    else 
        post_barre = [1,1,1]
        res = [34.0,34.0,34.0]
        for bloc in 1:n_bloc
            if state_list[3*(bloc-1)+1] == 3 || state_list[3*(bloc-1)+2] == 3
                post_barre[bloc] = 0
            end
        end
        for bloc in 1:n_bloc
            if state_list[3*(bloc-1)+3] == 3 #busbar failed
                res[bloc] = 0
            end
        end

        
        nb_working_barre = sum(post_barre)
        prod_tot = sum(res)
        for b in 1:n_bloc
            if res[b] > 0
                res[b] = prod_tot / nb_working_barre # on répartit la production 
            end
        end
    end
    return res
end

P_daily = [P_tp,P_dis,P_barre,P_tp,P_dis,P_barre,P_tp,P_dis,P_barre]
P_period = [P_tp^60,P_dis^60,P_barre^60,P_tp^60,P_dis^60,P_barre^60,P_tp^60,P_dis^60,P_barre^60]
