global model = x -> [0.0 for i in 1:(C+1)]

#include("learning.jl")
#include("borne_inf.jl")
include("weakly_coupled2.jl")

using Test

@testset begin
    
    @load "p2.jld2"
    @load "delta2.jld2"
    H = 6

    x0 = [1, 1, 1, 1, 1, 1, 1]
    k0 = 6

    m_opt, κ_opt, cost = PLNE(H,C,x0,k0,1,δ,p,n)
    @test m_opt == [0, 0, 0, 0, 0, 0, 0]
    @test κ_opt == 0

    x0 = [1, 3, 1, 1, 1, 1, 1]
    m_opt, κ_opt, cost = PLNE(H,C,x0,k0,1,δ,p,n)
    @test m_opt[2] == 1
    @test m_opt[1] == 0
    @test κ_opt > 0

    k0 = 0
    m_opt, κ_opt, cost = PLNE(H,C,x0,k0,1,δ,p,n)
    @test m_opt[2] == 1
    @test κ_opt == 0

    x0 = [1, 2, 7, 1, 2, 2, 2]
    k0 = 6
    m_opt, κ_opt, cost = PLNE(H,C,x0,k0,6,δ,p,n)
    @test m_opt[2] == 1
    @test κ_opt > 2

end

#= # test PLNE_borne_inf

@testset begin
    
    @load "p_30.jld2"
    @load "delta_30_inf.jld2"
    H = 6

    x0 = [1, 1, 1, 1, 1, 1, 1]
    k0 = 6

    m_opt, κ_opt, cost = PLNE_borne_inf(H,C,x0,k0,1,γ,p,n)
    @test m_opt == [0, 0, 0, 0, 0, 0, 0]
    @test κ_opt == 0

    x0 = [1, 3, 1, 1, 1, 1, 1]
    m_opt, κ_opt, cost = PLNE_borne_inf(H,C,x0,k0,1,γ,p,n)
    @test m_opt[2] == 1
    @test κ_opt > 0

    k0 = 0
    m_opt, κ_opt, cost = PLNE_borne_inf(H,C,x0,k0,1,γ,p,n)
    @test m_opt[2] == 1
    @test κ_opt == 0

    x0 = [1, 2, 7, 1, 2, 2, 2]
    k0 = 6
    m_opt, κ_opt, cost = PLNE_borne_inf(H,C,x0,k0,6,γ,p,n)
    @test m_opt[2] == 1
    @test κ_opt > 2

end

# test period cost vector 

# changer H en T

x_new = [1,1,1,1,1,1,1]
x_old = [12,1,1,1,1,1,1]
x_old_2 = [1,3,1,1,1,1,1]
x_old_3 = [1,2,4,1,1,1,1]
x_old_4 = [1,1,8,1,1,1,1]
theta_new = period_cost_vector(x_new,4,C,1)
theta_old = period_cost_vector(x_old,4,C,1)
theta_old_2 = period_cost_vector(x_old_2,4,C,1)
theta_old_2_sum = period_cost_vector(x_old_2,4,C,3)
theta_old_3 = period_cost_vector(x_old_3,4,C,1)
theta_old_4 = period_cost_vector(x_old_4,4,C,1)
println("cost of doing nothing when starting in new state: ",theta_new[(Q+1)*(1-1) + 1])
println("cost of doing nothing when component 1 is broken: ",theta_old[(Q+1)*(1-1) + 1])
println("cost of maintaining everything without quota when component 1 is broken: ",theta_old[(Q+1)*(4-1) + 1])
println("cost of maintaining everything with all quotas when component 1 is broken: ",theta_old[(Q+1)*(4-1) + 7])
println("cost of maintaining everything except 1 with all quotas when component 1 is broken: ",theta_old[(Q+1)*(3-1) + 7])
println("cost of maintaining everything except 1 without quotas when component 1 is broken: ",theta_old[(Q+1)*(3-1) + 1])
println("cost of doing nothing when component 2 is broken in winter: ",theta_old_2[(Q+1)*(1-1) + 1])
println("cost of maintenance of 1 day without quota when component 2 is broken in winter: ",theta_old_2[(Q+1)*(2-1) + 1])
println("cost of maintenance of 1 day with all quotas when component 2 is broken in winter: ",theta_old_2[(Q+1)*(2-1) + 7])
println("cost of maintenance of 1 day without quota when component 2 is broken in summer: ",theta_old_2_sum[(Q+1)*(2-1) + 1])
println("cost of maintenance of 1 day with all quotas when component 2 is broken in summer: ",theta_old_2_sum[(Q+1)*(2-1) + 7])
println("cost of maintenance of 1 day with all quotas when component 2 in state 2 and 3 in state 4 in winter: ",theta_old_3[(Q+1)*(2-1) + 7])
println("cost of doing nothing when component 2 in state 2 and 3 in state 4 in winter: ",theta_old_3[(Q+1)*(1-1) + 1])
println("cost of doing nothing when component 3 in state 8 in winter: ",theta_old_4[(Q+1)*(1-1) + 1])
println("cost of maintenance of 1 day with all quotas when component 3 in state 8 in winter: ",theta_old_4[(Q+1)*(2-1) + 7])

x_old_3 = [1,1,6,1,1,1,1]
theta_old_3 = period_cost_vector(x_old_3,4,C,1)
println("cost of maintenance of 1 day with all quotas when component 3 is in state 6 in winter: ",theta_old_3[(Q+1)*(2-1) + 7])
println("cost of doing nothing when component 3 is in state 6 in winter: ",theta_old_3[(Q+1)*(1-1) + 1])

# test full solution

@testset "full solution" begin

    # test that if we go from state decision to full solution and then back to state decision, we retrieve the same thing

    x = [1, 2, 3, 3, 2, 1, 1]
    k = 4
    m = [0, 1, 1, 0, 1, 1, 1]
    κ = 2
    y = full_solution(x,k,m,κ,L,C,Q)

    κ_opt = sum([(κ-1)*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
    κ_opt = Int(round(κ_opt))

    l_opt = sum([length_m[l]*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
    m_opt = [0.0 for i in 1:C]
    for c in 1:C
        if d[c] <= l_opt
            m_opt[c] = 1.0
        end
    end

    @test κ == κ_opt
    @test m == m_opt

end

# test CO layer 

@testset "CO layer" begin

    L = 4

    x = [1, 1, 1, 1, 1, 1, 1]
    k = 6

    θ_start = period_cost_vector(x,L,C,1)
    output = [0.0 for i in 1:(C+1)]
    θ = [-θ_start;output]
    y = CO_layer(θ; x, k, L, Q)

    κ_opt = sum([(κ-1)*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
    κ_opt = Int(round(κ_opt))

    l_opt = sum([length_m[l]*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
    m_opt = [0.0 for i in 1:C]
    for c in 1:C
        if d[c] <= l_opt
            m_opt[c] = 1.0
        end
    end

    @test κ_opt == 0
    @test m_opt == [0.0 for c in 1:7]

    # ---------------------------

    x = [1, 3, 1, 1, 1, 1, 1]
    k = 0

    θ_start = period_cost_vector(x,L,C,1)
    output = [0.0 for i in 1:(C+1)]
    θ = [-θ_start;output]
    y = CO_layer(θ; x, k, L, Q)

    κ_opt = sum([(κ-1)*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
    κ_opt = Int(round(κ_opt))

    l_opt = sum([length_m[l]*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])

    @test κ_opt == 0
    @test l_opt == 1

    # ---------------------------

    k = 6

    θ_start = period_cost_vector(x,L,C,1)
    output = [0.0 for i in 1:(C+1)]
    θ = [-θ_start;output]
    y = CO_layer(θ; x, k, L, Q)

    κ_opt = sum([(κ-1)*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
    κ_opt = Int(round(κ_opt))

    l_opt = sum([length_m[l]*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])

    @test κ_opt == 6
    @test l_opt == 1

    # ---------------------------

    x = [1, 1, 3, 1, 1, 1, 1]
    k = 6

    θ_start = period_cost_vector(x,L,C,1)
    output = [0.0 for i in 1:(C+1)]
    θ = [-θ_start;-output]
    y = CO_layer(θ; x, k, L, Q)

    κ_opt = sum([(κ-1)*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
    κ_opt = Int(round(κ_opt))

    l_opt = sum([length_m[l]*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
    m_opt = [0.0 for i in 1:C]
    for c in 1:C
        if d[c] <= l_opt
            m_opt[c] = 1.0
        end
    end

    @test κ_opt > 0
    @test m_opt == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # ---------------------------

    x = [1,2,4,1,1,1,1]

    k = 6

    θ_start = period_cost_vector(x,L,C,1)
    output = [0.0 for i in 1:(C+1)]
    θ = [-θ_start;output]
    y = CO_layer(θ; x, k, L, Q)

    κ_opt = sum([(κ-1)*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
    κ_opt = Int(round(κ_opt))

    l_opt = sum([length_m[l]*y[(l-1)*(Q+1) + κ] for l in 1:L, κ in 1:(Q+1)])
    m_opt = [0.0 for i in 1:C]
    for c in 1:C
        if d[c] <= l_opt
            m_opt[c] = 1.0
        end
    end

    @test κ_opt > 0
    @test m_opt == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]


end

# test loss 

@testset "loss" begin

    perturbed_maximizer = PerturbedAdditive(
        CO_layer; ε=0.1, nb_samples=20, seed=1
    )
    loss = FenchelYoungLoss(perturbed_maximizer)

    L = 4
    T = 1

    x = [1 for i in 1:C]
    k = 6
    m = [0.0 for c in 1:C]
    κ = 0
    y = full_solution(x,k,m,κ,L,C,Q)
        
    θ_start = period_cost_vector(x,L,C,T)
    output = [0.0 for i in 1:(C+1)]
    θ = [-θ_start;-K_mult*output]

    l = loss(θ, y; x, k, L, Q)
    tol = 1
    @test l > -tol

    # ---------------------------

    m = [0, 1, 1, 0, 1, 1, 1]
    y = full_solution(x,k,m,κ,L,C,Q)
    l = loss(θ, y; x, k, L, Q)
    @test l > 100

    # ---------------------------

    κ = 6
    output = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10000]
    θ = [-θ_start;-K_mult*output]
    y = full_solution(x,k,m,κ,L,C,Q)
    l = loss(θ, y; x, k, L, Q)
    @test l > 100

end



# test benchmark policy

@testset "benchmark policy" begin

    x = [1, 2, 8, 1, 1, 1, 1]
    k = 6

    @load "delta01monopole.jld2"
    @load "p01monopole.jld2"

    Policies_m_t_list, Policies_κ_t = π_learning(x,k,1,h[1,1,:],"policies_l.csv",δ,p,pr[1,:],n,d,P_daily01)

    @test sum(Policies_m_t_list[3]) > 0

end
 =#