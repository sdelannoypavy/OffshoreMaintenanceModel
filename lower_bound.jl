# lower_bound.jl
#
# Borne inférieure sur le coût optimal sur l'horizon complet,
# en résolvant le LP fluide de weakly_coupled3.jl UNE SEULE FOIS
# depuis l'état initial (tous les composants neufs, quota plein, t=1).

using JLD2, Gurobi, JuMP, Printf

# ── Paramètres utilisateur ────────────────────────────────────────────────────
years  = 20    # durée de la concession en années
Q      = 6     # quota annuel de jours de maintenance libres

delta_file = "delta_lb.jld2"
p_file     = "p_lb.jld2"

# ── Chargement du problème (7 composants) ────────────────────────────────────
include("get_scenarios.jl")
include("create_scenarios_prod.jl")
include("build_MC_utils.jl")
include("create_simple_pole.jl")        # C=7, n, d, nb_time=4, length_m, P_daily, P_evac
include("parameters_weakly_coupled.jl") # create_p, average_period_cost, create_schedule_*

H_per = 6

include("lb_data_utils.jl")

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

@load delta_file δ
@load p_file p

@assert ndims(δ) == 5              "δ doit avoir 5 dimensions"
@assert size(δ, 3) == nb_time      "δ dim 3 ≠ nb_time=$(nb_time)"
@assert size(δ, 4) == Q + 1        "δ dim 4 ≠ Q+1=$(Q+1)"
@assert ndims(p) == 8              "p doit avoir 8 dimensions"
@assert size(p, 6) == 2            "p dim 6 ≠ 2 (binaire m) : relancer lower_bound.jl"

H_full = 6 * years

# ── Paramètres spécifiques à weakly_coupled3 ─────────────────────────────────
int_m = [3, 1, 1, 2, 1, 1, 1]   # fenêtres individuelles max par composant

x0 = ones(Int, C)
k0 = Q

@printf("\n%s\n", "="^65)
@printf("BORNE INFÉRIEURE — LP weakly_coupled3, horizon complet\n")
@printf("%s\n", "-"^65)
@printf("  C=%d  n_max=%d  Q=%d  nb_time=%d  H=%d étapes (%d ans)\n",
        C, n_max, Q, nb_time, H_full, years)
@printf("  length_m = %s\n", string(length_m))
@printf("%s\n", "="^65)

# ── Gurobi ────────────────────────────────────────────────────────────────────
const GRB_ENV = Gurobi.Env()
model = Model(() -> Gurobi.Optimizer(GRB_ENV))
set_optimizer_attribute(model, "Method",    2)
set_optimizer_attribute(model, "Crossover", 0)
set_optimizer_attribute(model, "LogFile",   "lower_bound_gurobi.log")

# ── Variables ─────────────────────────────────────────────────────────────────
@variable(model, q[c=1:C, x=1:n[c], k=0:Q, m=1:2, κ=0:Q, t=1:H_full] >= 0)  # m=1: non maintenu, m=2: maintenu
@variable(model, A[κ=0:Q, m=1:nb_time, t=1:H_full] >= 0)
@variable(model, μ[k=0:Q, t=1:H_full] >= 0)

# ── Objectif (weakly_coupled3 avec indexation cyclique) ──────────────────────
# Partie 1 : coûts individuels, m ≤ int_m[c]
# Partie 2 : coûts de groupe pour fenêtres m=2,3,4
#   (limités à nb_time pour éviter les accès hors bornes sur δ)
part2_terms = AffExpr()
if nb_time >= 2; add_to_expression!(part2_terms, sum(δ[2,1,2,κ+1,(t-1)%6+1]*A[κ,2,t] for κ in 0:Q, t in 1:H_full)); end
if nb_time >= 3; add_to_expression!(part2_terms, sum(δ[4,1,3,κ+1,(t-1)%6+1]*A[κ,3,t] for κ in 0:Q, t in 1:H_full)); end
if nb_time >= 4; add_to_expression!(part2_terms, sum(δ[1,1,4,κ+1,(t-1)%6+1]*A[κ,4,t] for κ in 0:Q, t in 1:H_full)); end

@objective(model, Min,
    # Partie 1 : coûts de panne (m=1, pas de maintenance programmée)
    sum(δ[c,x,1,κ+1,(t-1)%6+1] * q[c,x,k,1,κ,t]
        for c in 1:C, x in 1:n[c], k in 0:Q, κ in 0:Q, t in 1:H_full)
    # Partie 2 : coût de réduction de capacité pendant la fenêtre de maintenance (via A)
    + part2_terms
    # Partie 3 : coût fixe par composant × probabilité d'être en maintenance (m=2)
    + sum(comp_cost[c] *
          sum(q[c,x,k,2,κ,t] for x in 1:n[c], k in 0:Q, κ in 0:Q, t in 1:H_full)
          for c in 1:min(C, length(comp_cost))))

# ── Contraintes ───────────────────────────────────────────────────────────────

@constraint(model, flow[c=1:C, t=2:H_full, x=1:n[c], k=0:Q],
    sum(q[c,x,k,m,κ,t] for m in 1:2, κ in 0:Q)
    >=
    sum(p[c,x′,k′+1,x,k+1,m,κ+1,(t-2)%6+1] * q[c,x′,k′,m,κ,t-1]
        for x′ in 1:n[c], k′ in 0:Q, m in 1:2, κ in 0:Q))

# Couplage : prob de maintenance de c ≤ somme des fenêtres de durée ≥ d[c]
@constraint(model, coupling[c=1:C, κ=0:Q, t=1:H_full],
    sum(q[c,x,k,2,κ,t] for x in 1:n[c], k in 0:Q)
    <= sum(A[κ,m,t] for m in 1:nb_time if length_m[m] >= d[c]))

α = zeros(C, n_max, Q+1)
for c in 1:C; α[c, x0[c], k0+1] = 1.0; end
@constraint(model, init[c=1:C, x=1:n[c], k=0:Q],
    sum(q[c,x,k,m,κ,1] for m in 1:2, κ in 0:Q) == α[c,x,k+1])

@constraint(model, mono[c=1:C, x=1:n[c], k=0:(Q-1), m=1:2, κ=(k+1):Q, t=1:H_full],
    q[c,x,k,m,κ,t] == 0)

@constraint(model, quota[c=1:C, k=0:Q, t=1:H_full],
    sum(q[c,x,k,m,κ,t] for x in 1:n[c], m in 1:2, κ in 0:Q) == μ[k,t])

# ── Résolution ────────────────────────────────────────────────────────────────
@printf("\nConstruction terminée. Résolution en cours...\n")
flush(stdout)

optimize!(model)

status = termination_status(model)
lb     = objective_value(model)

@printf("\n%s\n", "="^65)
@printf("RÉSULTATS\n")
@printf("%s\n", "-"^65)
@printf("  Statut solveur                                : %s\n", status)
@printf("  Borne inférieure (LP fluide, horizon complet) : %.4f\n", lb)
@printf("  Borne par année (÷ %d ans)                    : %.4f\n", years, lb / years)

# ── Politique optimale extraite au pas t=1 ────────────────────────────────────
κ_star = begin
    mass_κ = zeros(Q+1)
    for c in 1:C, x in 1:n[c], k in 0:Q, m in 1:2, κ in 0:Q
        mass_κ[κ+1] += value(q[c,x,k,m,κ,1])
    end
    argmax(mass_κ) - 1
end

m_star = begin
    _m_star = zeros(Int, C)
    for c in 1:C
        mass_m = zeros(2)
        for x in 1:n[c], k in 0:Q, κ in 0:Q, m in 1:2
            mass_m[m] += value(q[c,x,k,m,κ,1])
        end
        _m_star[c] = argmax(mass_m) >= 2 ? 1 : 0
    end
    _m_star
end

@printf("%s\n", "-"^65)
@printf("  Politique optimale au pas t=1\n")
@printf("  κ* (jours libres)  : %d\n", κ_star)
@printf("  m* par composant   : %s\n", string(m_star))
@printf("%s\n", "="^65)
