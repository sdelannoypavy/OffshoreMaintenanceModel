C = 3

#transformateur de puissance, disjoncteur, transformateur de mesure
d = [1, 1, 1] 
n = [4, 4, 4] # number of blocs still working 

MTBF_tp = 2000
MTBF_dis = 20
MTBF_tm = 40

p_tp = 1/(365*MTBF_tp) # we have the MTBF for one transformer 
p_dis = 1/(365*MTBF_dis)
p_tm = 1/(365*MTBF_tm)

P_tp = [1 - 3*p_tp*(1- p_tp)^2  0                   0                0
        3*p_tp*(1- p_tp)^2      1 - 2*p_tp*(1-p_tp) 0                0
        0                       2*p_tp*(1-p_tp)     1 - p_tp         0
        0                       0                   p_tp             1]

P_dis = [1 - 3*p_dis*(1- p_dis)^2  0                     0                0
        3*p_dis*(1- p_dis)^2       1 - 2*p_dis*(1-p_dis) 0                0
        0                          2*p_dis*(1-p_dis)     1 - p_dis        0
        0                          0                     p_dis            1]


P_tm = [1 - 3*p_tm*(1- p_tm)^2  0                   0                0
        3*p_tm*(1- p_tm)^2      1 - 2*p_tm*(1-p_tm) 0                0
        0                       2*p_tm*(1-p_tm)     1 - p_tp         0
        0                       0                   p_tp             1]

comp_cost = [1.86*100, 0.1*100, 0.01*100] #cost of maintenance for each component

n_max = maximum(n) #we can create arrays with irregular size

P_evac = [100.0 for c in 1:C, x in 1:n_max]
for c in 1:C
    P_evac[c,1:4] = [100.0, 66.0, 33.0, 0.0]
end

#linear decrease for the fan

function Capacity(state_list,n,C,P_evac,q)
    if sum(q[c] for c in 1:C) >= 1 #at least one ongoing maintenance
        return 0
    else 
        return minimum([P_evac[c,state_list[c]] for c in 1:C])
    end
end

P_daily = [P_tp,P_dis,P_tm]
P_period = [P_tp^60,P_dis^60,P_tm^60]
