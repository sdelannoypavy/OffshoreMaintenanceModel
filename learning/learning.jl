#the neural network takes as input [x0; k0; t0]
#the neural network outputs [θ_m; θ_κ] where θ_m of length c is associated to component states (should be positive)
#and θ_κ associated to remaining maintenance days (should be negative)
#the cost vector of also contains costs over the scheduling stage associated to scheduling decisions
#those costs are computed when the CO layer is called because there are two many costs to be computed beforehand


using Plots
using Flux
using InferOpt
using Random
using Revise
using CSV
using JLD2
using JuMP
using Gurobi
using DataFrames

L= 4
Q = 6
nb_scen = 80

K_mult = 100
    

include("CO_layer.jl")
include("create_simple_pole.jl")
include("get_scenarios.jl")
include("create_scenarios_prod.jl")
include("simulation multiple comps.jl")
include("parameters_weakly_coupled.jl")

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



function main()

    # Hyperparameters
    lr_start = 0.001
    nb_samples = 20 # number of perturbed samples
    ε = 10.0  # scale of the perturbation
    Nb_epochs = 1
    training_data_nb = 5000

    C = 7
    L = 4 
    Q = 6 
    nb_scen = 80

    seed = 19

    learning_dim = L*(Q+1) + C + 1

    train_data = create_data(training_data_nb)

    #train setup
    
    perturbed_maximizer = PerturbedAdditive(
        CO_layer; ε=ε, nb_samples=nb_samples, seed=seed
    )
    loss = FenchelYoungLoss(perturbed_maximizer)

    model = Chain(
    Dense(C + 2 => 2C + 2, tanh; bias=true),   
    Dense(2C + 2 => C + 2, tanh; bias=true),   
    Dense(C + 2 => C + 1; bias=true)          
    )

    opt = Adam(lr_start)

    loss_list = []
    l= 0.0
    for (x, k, T, m, κ) in train_data

        input = [x; k; T]
        y = [m; κ]
    
        output = model(input)
        y = full_solution(x,k,m,κ,L,C,Q)
        @assert length(y) == learning_dim
        
        θ_start = period_cost_vector(x,L,C,T)
        # negative because maximization problem
        θ = [-θ_start;-K_mult*output]
        @assert length(θ) == learning_dim
        
        l += loss(θ, y; x, k, L, Q)
    end
    push!(loss_list, l)
    println("Loss: "*string(l))


    for _ in 1:Nb_epochs #for the moment it it just times when we stop to plot the loss
        model, l = train_model(
        model,
        train_data;
        opt,
        loss,
        L,
        Q,
        nb_scen
    )
        push!(loss_list, l)

        println("Loss: "*string(l))


    end

    x = 0:Nb_epochs
    plot(x,loss_list)

    savefig("learning.pdf")

    return model

end

function train_model(
    model,
    train_data;
    opt,
    loss,
    L,
    Q,
    nb_scen
)
    
    opt_state = Flux.setup(opt, model)
    losses = Float64[]

    l= 0.0
    for (x, k, T, m, κ) in train_data

        input = [x; k; T]
        y = full_solution(x,k,m,κ,L,C,Q)
        @assert length(y) == learning_dim
        
        #Flux.reset!(model) # init hidden state, not sure it is usefull
        θ_start = period_cost_vector(x,L,C,T)

        output = model(input)
        θ = [-θ_start;-K_mult*output]
        #dot_val = dot(θ,y)
        #println("true dot: $dot_val")

        grads = Flux.gradient(model) do m
            output = m(input)
            θ = [-θ_start;-K_mult*output]
            l += loss(θ, y; x, k, L, Q)        
        end
        Flux.update!(opt_state, model, grads[1])
    
    end

    return model, l
end

function create_data(training_data_nb)

    df1 = CSV.read("policies_sensitivity201.csv", DataFrame)
    df2 = CSV.read("policies_sensitivity299.csv", DataFrame)

    train_data = []

    for tr in 1:training_data_nb

        j = rand(1:2)
        if j == 1
            df = df1
        else
            df = df2
        end

        n = rand(1:500)

        x = [df[n,c] for c in 1:C] # states
        k = df[n, C+1]
        T = df[n, C+2]
        m = [df[n, C+2+c] for c in 1:C]
        κ = df[n, 17]

        push!(train_data, (x, k, T, m, κ))
    end

    return train_data

end

function testing()
    nb_w = 500
    nb_d = 1
    years = 30
    H = 6

    @load "delta01monopole.jld2"
    δ01 = δ
    @load "delta99monopole.jld2"
    δ99 = δ
    #@load "delta67monopole.jld2"
    #δ67 = δ

    @load "p01monopole.jld2"
    p01 = p
    @load "p99monopole.jld2"
    p99 = p
    #@load "p67monopole.jld2"
    #p67 = p

    cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()
    open("policies_l.csv", "w") do f
        write(f, "")
    end

    open("simulationl01.csv", "w") do f
        write(f, "")
    end

    open("simulationl99.csv", "w") do f
        write(f, "")
    end

    #open("simulationl67.csv", "w") do f
    #    write(f, "")
    #end

    #simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily01,"simulationl01.csv","policies_l.csv","learning",δ01,p01,"costl01.csv",false,C)
    #cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()
    simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily99,"simulationl99.csv","policies_l.csv","learning",δ99,p99,"costl99.csv",false,C)
    #cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()
    #simulate(nb_w,nb_d,H,h,pr,n,years,Q,d,P_daily67,"simulationl67.csv","policies_l2.csv","learning",δ67,p67,"costl67.csv",false,C)

end

#model = main()
#@save "model3.jld2" model
@load "model3.jld2"
testing()

# training is not right because the objective is not linear in maintenance schedule. 
# the CO layer should ouput the expectation of states and k_end
# but still it is not additive in theta because we have the cost over scheduling stage 
# so theta should include some components that are not outputed by nn ? 
# the global objective is linear. 

#loss doesnt converge to 0 with dataset of size 1: bug? 

# change value of K_mult before testing 


#df = CSV.read("costl99.csv", DataFrame)
#mean(df[:, 2])

# calcul takes time maybe because we have to compute the cost vector each time (because state = product state)