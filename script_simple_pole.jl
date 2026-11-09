# script_simple_pole.jl
#
# Simulation de la politique fluide (structure LP de weakly_coupled3)
# pour la station simple (7 composants), avec coûts fixes de maintenance.
# Les fichiers delta_lb.jld2 et p_lb.jld2 doivent avoir été générés
# par lower_bound.jl avant de lancer ce script.

using JLD2, Gurobi, JuMP, Printf, StatsBase, CSV, DataFrames, Statistics

include("get_scenarios.jl")
include("create_scenarios_prod.jl")
include("build_MC_utils.jl")
include("create_simple_pole.jl")        # C=7, n, d, nb_time=4, length_m, P_daily, P_evac, comp_cost
include("parameters_weakly_coupled.jl") # create_schedule_m, create_schedule_κ, t_start, margin
include("lb_data_utils.jl")             # create_δ_full, create_p_full, _need_recompute_*
include("simulation_utils.jl")          # simulate_period, simulate_concession

# ── Paramètres ────────────────────────────────────────────────────────────────
years  = 20
Q      = 6
nb_w   = 200   # nombre de scénarios météo simulés
nb_d   = 1     # tirages de dégradation par scénario météo
H      = 6     # horizon de planification roulant (en périodes)

delta_file    = "delta_lb.jld2"
p_file        = "p_lb.jld2"
policies_file = "policies_simpole.csv"
simu_file     = "simulation_simpole.csv"
cost_file     = "cost_simpole.csv"

int_m = [3, 1, 1, 2, 1, 1, 1]   # fenêtres individuelles max par composant

# ── Calcul de δ et p si nécessaire ───────────────────────────────────────────
H_per = 6

if _need_recompute_δ(delta_file, nb_time, Q)
    isfile(delta_file) && (@printf("  δ existant invalide → suppression\n"); rm(delta_file))
    @printf("Calcul de δ → %s\n", delta_file)
    create_δ_full(C, h_optim, pr, n, d, H_per, P_daily, delta_file, nb_time, P_evac)
end

if _need_recompute_p(p_file, nb_time, Q)
    isfile(p_file) && (@printf("  p existant invalide → suppression\n"); rm(p_file))
    @printf("Calcul de p → %s\n", p_file)
    create_p_full(C, H_per, n, P_period, p_file, nb_time)
end

# ── Chargement de δ et p ──────────────────────────────────────────────────────
@load delta_file δ
@load p_file p

@assert size(δ, 3) == nb_time "δ incompatible"
@assert size(p, 6) == 2       "p incompatible (dim 6 ≠ 2)"

# ── Gurobi ────────────────────────────────────────────────────────────────────
const GRB_ENV = Gurobi.Env()
optimizer_fn  = () -> Gurobi.Optimizer(GRB_ENV)

# ── Cache des politiques ───────────────────────────────────────────────────────
cache = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64}, Tuple{Vector{Int64}, Int64}}()

function collect_policies(file_name)
    !isfile(file_name) && return
    df = CSV.read(file_name, DataFrame; header=false)
    for i in 1:nrow(df)
        v     = collect(df[i, :])
        x0    = Int.(v[1:C])
        k0    = Int(v[C+1])
        t0    = Int(v[C+2])
        m_opt = Int.(v[(C+3):(2C+2)])
        κ_opt = Int(v[2C+3])
        cache[(Tuple(x0), k0, t0)] = (m_opt, κ_opt)
    end
end

# ── LP roulant ────────────────────────────────────────────────────────────────
# Même structure d'objectif que lower_bound.jl :
#   Partie 1 : coûts de panne (m=1 uniquement)
#   Partie 2 : coûts de maintenance fixes (m≥2, via A)
# p est indexé directement (t-1) car Tmax = min(H+t0-1, 12) ≤ 12.

function PLNE_simpole(x0, k0, t0)
    Tmax = min(H + t0 - 1, 12)

    α = zeros(C, n_max, Q+1)
    for c in 1:C; α[c, x0[c], k0+1] = 1.0; end

    model = Model(optimizer_fn)
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, q[c=1:C, x=1:n[c], k=0:Q, m=1:2, κ=0:Q, t=t0:Tmax] >= 0)  # m=1: non maintenu, m=2: maintenu
    @variable(model, A[κ=0:Q, m=1:nb_time, t=t0:Tmax] >= 0)
    @variable(model, μ[k=0:Q, t=t0:Tmax] >= 0)

    part2 = AffExpr()
    if nb_time >= 2; add_to_expression!(part2, sum(δ[2,1,2,κ+1,t]*A[κ,2,t] for κ in 0:Q, t in t0:Tmax)); end
    if nb_time >= 3; add_to_expression!(part2, sum(δ[4,1,3,κ+1,t]*A[κ,3,t] for κ in 0:Q, t in t0:Tmax)); end
    if nb_time >= 4; add_to_expression!(part2, sum(δ[1,1,4,κ+1,t]*A[κ,4,t] for κ in 0:Q, t in t0:Tmax)); end

    @objective(model, Min,
        sum(δ[c,x,1,κ+1,t] * q[c,x,k,1,κ,t]
            for c in 1:C, x in 1:n[c], k in 0:Q, κ in 0:Q, t in t0:Tmax)
        + part2
        + sum(comp_cost[c] *
              sum(q[c,x,k,2,κ,t] for x in 1:n[c], k in 0:Q, κ in 0:Q, t in t0:Tmax)
              for c in 1:min(C, length(comp_cost))))

    @constraint(model, [c=1:C, t=(t0+1):Tmax, x=1:n[c], k=0:Q],
        sum(q[c,x,k,m,κ,t] for m in 1:2, κ in 0:Q)
        >= sum(p[c,x′,k′+1,x,k+1,m,κ+1,t-1] * q[c,x′,k′,m,κ,t-1]
               for x′ in 1:n[c], k′ in 0:Q, m in 1:2, κ in 0:Q))

    # Couplage : prob de maintenance de c ≤ somme des fenêtres de durée ≥ d[c]
    @constraint(model, [c=1:C, κ=0:Q, t=t0:Tmax],
        sum(q[c,x,k,2,κ,t] for x in 1:n[c], k in 0:Q)
        <= sum(A[κ,m,t] for m in 1:nb_time if length_m[m] >= d[c]))

    @constraint(model, [c=1:C, x=1:n[c], k=0:Q],
        sum(q[c,x,k,m,κ,t0] for m in 1:2, κ in 0:Q) == α[c,x,k+1])

    @constraint(model, [c=1:C, x=1:n[c], k=0:(Q-1), m=1:2, κ=(k+1):Q, t=t0:Tmax],
        q[c,x,k,m,κ,t] == 0)

    @constraint(model, [c=1:C, k=0:Q, t=t0:Tmax],
        sum(q[c,x,k,m,κ,t] for x in 1:n[c], m in 1:2, κ in 0:Q) == μ[k,t])

    optimize!(model)

    # κ_opt : quota dominant via A (inchangé)
    κ_opt = argmax([sum(value(A[κ,m,t0]) for m in 1:nb_time) for κ in 0:Q]) - 1

    # m_opt[c] : 1 si la masse en m=2 (maintenu) domine, 0 sinon
    m_opt = begin
        _m_opt = zeros(Int, C)
        for c in 1:C
            mass_m = zeros(2)
            for x in 1:n[c], k in 0:Q, κ in 0:Q, m in 1:2
                mass_m[m] += value(q[c,x,k,m,κ,t0])
            end
            _m_opt[c] = argmax(mass_m) >= 2 ? 1 : 0
        end
        _m_opt
    end

    return m_opt, κ_opt
end

# ── Politique fluide avec cache ───────────────────────────────────────────────
# Interface : (x0, k0, t_global, h_t, pr_t) → (Policies_m_t_list, Policies_κ_t)
function policy_fluid(x0, k0, t_global, h_t, pr_t)
    T_per = 62 - count(==(-1), h_t)
    t0    = (t_global - 1) % 6 + 1
    cle   = (Tuple(x0), k0, t0)

    if haskey(cache, cle)
        m_opt, κ_opt = cache[cle]
    else
        m_opt, κ_opt = PLNE_simpole(x0, k0, t0)
        cache[cle]   = (m_opt, κ_opt)
        row = DataFrame(reshape([x0; k0; t0; m_opt; κ_opt], 1, :), :auto)
        CSV.write(policies_file, row; append=true, writeheader=false)
    end

    if sum(m_opt) == 0
        Policies_m_t_list = [create_schedule_m(0, 1, T_per) for c in 1:C]
        Policies_κ_t      = create_schedule_κ(κ_opt, 1, T_per)
        return Policies_m_t_list, Policies_κ_t
    end

    cost_min          = Inf
    Policies_m_t_best = [create_schedule_m(m_opt[c], 1, T_per) for c in 1:C]
    Policies_κ_t_best = create_schedule_κ(κ_opt, 1, T_per)

    for t_s in 1:(60 - margin)
        Pols_m = [create_schedule_m(m_opt[c], t_s, T_per) for c in 1:C]
        Pols_κ = create_schedule_κ(κ_opt, t_s, T_per)
        cost, _ = simulate_period(Pols_m, Pols_κ, copy(x0), h_t, pr_t)
        if cost < cost_min
            cost_min          = cost
            Policies_m_t_best = Pols_m
            Policies_κ_t_best = Pols_κ
        end
    end

    return Policies_m_t_best, Policies_κ_t_best
end

# ── Boucle principale ─────────────────────────────────────────────────────────
collect_policies(policies_file)

@printf("\n%s\n", "="^65)
@printf("SIMULATION — politique fluide, station simple, %d ans\n", years)
@printf("  C=%d  Q=%d  nb_w=%d  H=%d\n", C, Q, nb_w, H)
@printf("%s\n", "="^65)

costs_all = Float64[]

for scen_w in 1:nb_w
    h_sim = zeros(Int, 6*years, 62)
    for t in 1:(6*years)
        t_cyc      = (t-1) % size(h,1) + 1
        s          = rand(1:size(h,2))
        h_sim[t,:] = h[t_cyc, s, :]
    end

    cost_mean = 0.0
    for _ in 1:nb_d
        cost_mean += simulate_concession(policy_fluid, h_sim, pr, Q, years,
                                             comp_cost, simu_file)
    end
    cost_mean /= nb_d
    push!(costs_all, cost_mean)

    row = DataFrame(reshape(["fluid_simpole"; cost_mean], 1, :), :auto)
    CSV.write(cost_file, row; append=true, writeheader=false)

    if scen_w % 20 == 0
        @printf("  scénario %3d/%d — coût moyen : %.4f\n", scen_w, nb_w, mean(costs_all))
    end
end

@printf("\n%s\n", "="^65)
@printf("  Coût total moyen (sur %d ans)  : %.4f\n", years, mean(costs_all))
@printf("  Coût annuel moyen              : %.4f\n", mean(costs_all) / years)
@printf("  Écart-type                     : %.4f\n", std(costs_all))
@printf("%s\n", "="^65)
