# every MTBF (in years) is multiplied by MTBF_SCALE (environment variable, default 1.0,
# e.g. `MTBF_SCALE=0.5 julia ...` halves every MTBF)
MTBF_SCALE = parse(Float64, get(ENV, "MTBF_SCALE", "1.0"))
MTBF_conv = 588 * MTBF_SCALE
MTBF_water = 43 * MTBF_SCALE
MTBF_fan = 43 * MTBF_SCALE
MTBF_pump = 43 * MTBF_SCALE
MTBF_trans = 2222 * MTBF_SCALE
MTBF_cool = 122 * MTBF_SCALE
MTBF_bus = 4762 * MTBF_SCALE

C = 7
#components: converter, water circuit, fans, pumps, transformer, cooling, busbar
n = [12, 3, 8, 3, 3, 3, 3]


function build_P_linear_daily(nb_state,MTBF)

    p = ((nb_state - 1)/MTBF)/365

    P = [1 - p    0       0;
               p        1 - p   0;
               0        p       1]

end


function build_P_converter(MTBF,nb_state)

    p = (11/MTBF)/365

    P = zeros(nb_state, nb_state)
        
    for i in 1:(nb_state - 1)
        P[i,i] = 1.0 - p 
        P[i+1,i] = p
    end
        
    P[nb_state,nb_state] = 1.0

    return P
end

function build_P_fan(MTBF)

    p = (7/MTBF)/365

    nb_state = n[3]

    P = zeros(nb_state, nb_state)
        
    for i in 1:(nb_state - 1) # should be binomial, but we model like this for simplicity
        P[i+1,i] = p
        P[i,i] = 1.0 - p
    end
        
    P[nb_state,nb_state] = 1.0

    return P
end

p_pump = 9.55e-5 / MTBF_SCALE # gives MTBF_pump around 43 * MTBF_SCALE
P_pump = [1 - 2*p_pump*(1- p_pump)  0                 0
          2*p_pump*(1- p_pump)      1 - p_pump        0
          0                         p_pump            1]

p_trans = 1/(365*MTBF_trans) # we have the MTBF for one transformer 
P_trans = [1 - 2*p_trans*(1- p_trans)  0                 0
          2*p_trans*(1- p_trans)      1 - p_trans        0
          0                         p_trans              1]
          
P_water = build_P_linear_daily(n[2],MTBF_water)
P_cool = build_P_linear_daily(n[6], MTBF_cool)
P_bus = build_P_linear_daily(n[7], MTBF_bus)
P_fan = build_P_fan(MTBF_fan)
P_conv = build_P_converter(MTBF_conv,n[1])

#verifier le MTBF de fan
#verifier les MTBF sur l'ensemble des composants
