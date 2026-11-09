# degradation_proba.jl
#
# Section 1 : coût simulé sur 2 mois — maintenance 1 jour, 6 quotas,
#             état initial tous neufs, en fonction du jour de début t ∈ 1:45
#
# Section 2 : pour chaque composant c :
#   a. Distribution d'état après 60 jours sans maintenance depuis l'état 1
#   b. Fonction valeur V(c, x) estimée par le LP fluide (horizon H=6 périodes)
#      en supposant que tous les autres composants j≠c sont dans l'état 1
#   c. Espérance de la fonction valeur : E[V(c)] = Σ_x P(état=x après 60j) × V(c,x)

using JLD2, Gurobi, JuMP, Printf

include("get_scenarios.jl")
include("create_scenarios_prod.jl")
include("build_MC_utils.jl")
include("create_simple_pole.jl")        # C, n, d, nb_time, length_m, P_daily, P_period, P_evac, comp_cost
include("parameters_weakly_coupled.jl") # t_start, create_schedule_m, create_schedule_κ
include("lb_data_utils.jl")             # create_δ_full, create_p_full, _need_recompute_*

# ── Paramètres ────────────────────────────────────────────────────────────────
Q       = 6
H       = 6
H_per   = 6

delta_file = "delta_lb.jld2"
p_file     = "p_lb.jld2"

comp_names = ["Convertisseur  ", "Circuit eau    ", "Ventilateur    ",
              "Pompe          ", "Transformateur ", "Refroidissement", "Jeu de barres  "]

# ── Écriture simultanée stdout + fichier ──────────────────────────────────────
const RESULTS_FILE = "degradation_results.txt"
global _io_log = open(RESULTS_FILE, "w")

function teeprintf(fmt::String, args...)
    s = Printf.format(Printf.Format(fmt), args...)
    print(stdout, s)
    print(_io_log, s)
end

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Coût simulé en fonction du jour de début de maintenance
# ══════════════════════════════════════════════════════════════════════════════

nb_scen_h = size(h_optim, 2)   # scénarios météo disponibles par période
t_per_ref = 1                  # période de référence (1 = premier bimestre)
κ_fixed   = 6                  # jours de quota utilisés
# composant représentatif pour average_period_cost : d[c]=1 pour vraie maintenance 1 jour
# c=2 (circuit eau, d=1, n=3, P_evac=100 en état 1)
c_ref     = 2

costs_vs_t = zeros(45)
for t_s in 1:45
    total = 0.0
    for scen in 1:nb_scen_h
        h_t   = h_optim[t_per_ref, scen, :]
        T_per = 62 - count(==(-1), h_t)
        Pols_m = create_schedule_m(1, t_s, T_per)
        Pols_κ = create_schedule_κ(κ_fixed, t_s, T_per)
        total += average_period_cost(T_per, Pols_m, Pols_κ,
                                     2,                      # état initial = état 2
                                     h_t, pr[t_per_ref, :],
                                     n[c_ref], d[c_ref],
                                     P_daily[c_ref], P_evac[c_ref, :])
    end
    costs_vs_t[t_s] = total / nb_scen_h
end

teeprintf("\n%s\n", "="^65)
teeprintf("SECTION 1 — Coût espéré vs jour de début de maintenance\n")
teeprintf("  Maintenance 1 jour, κ=%d quotas, composant 2 en état 2\n", κ_fixed)
teeprintf("  Moyenne sur %d scénarios météo (average_period_cost)\n", nb_scen_h)
teeprintf("%s\n", "="^65)
teeprintf("  %4s   %10s\n", "t", "Coût moyen")
teeprintf("  %s\n", "-"^20)
for t_s in 1:45
    teeprintf("  %4d   %10.4f\n", t_s, costs_vs_t[t_s])
end
t_best = argmin(costs_vs_t)
teeprintf("  %s\n", "-"^20)
teeprintf("  Jour optimal : t = %d  (coût = %.4f)\n", t_best, costs_vs_t[t_best])
teeprintf("%s\n", "="^65)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Distribution de dégradation + fonction valeur fluide
# ══════════════════════════════════════════════════════════════════════════════

# ── Calcul de δ et p si nécessaire ────────────────────────────────────────────
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

# ── Gurobi ────────────────────────────────────────────────────────────────────
const GRB_ENV = Gurobi.Env()
optimizer_fn  = () -> Gurobi.Optimizer(GRB_ENV)

# ── LP fluide : valeur optimale depuis x0, quota k0, période t0 ───────────────
function compute_value(x0, k0, t0)
    Tmax = min(H + t0 - 1, 12)
    α = zeros(C, n_max, Q+1)
    for c in 1:C; α[c, x0[c], k0+1] = 1.0; end

    model = Model(optimizer_fn)
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, q[c=1:C, x=1:n[c], k=0:Q, m=1:2, κ=0:Q, t=t0:Tmax] >= 0)
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
    return objective_value(model)
end

# ── Calcul section 2 ──────────────────────────────────────────────────────────
teeprintf("\n%s\n", "="^75)
teeprintf("SECTION 2 — Dégradation après 60 j + fonction valeur fluide")
teeprintf(" (H=%d périodes, k=%d)\n", H, Q)
teeprintf("Hypothèse : tous les autres composants dans l'état 1\n")
teeprintf("%s\n", "="^75)

for c in 1:C
    distrib   = P_period[c][1:n[c], 1]
    p_degrade = sum(distrib[2:end])

    V = zeros(n[c])
    for x in 1:n[c]
        x0    = ones(Int, C)
        x0[c] = x
        V[x]  = compute_value(x0, Q, 1)
    end

    EV = sum(distrib[x] * V[x] for x in 1:n[c])

    teeprintf("\n%s  (n=%d états)\n", comp_names[c], n[c])
    teeprintf("  Distribution après 60 j  : [%s]\n",
              join([@sprintf("%.4f", distrib[x]) for x in 1:n[c]], "  "))
    teeprintf("  P(dégradé)               : %.4f\n", p_degrade)
    teeprintf("  Fonction valeur V(c,x)   : [%s]\n",
              join([@sprintf("%.3f", V[x]) for x in 1:n[c]], "  "))
    teeprintf("  E[V] après 60 j          : %.4f  (vs V(c,1) = %.4f,  Δ = %.4f)\n",
              EV, V[1], EV - V[1])
end

teeprintf("\n%s\n", "="^75)

# ── Fermeture du fichier ───────────────────────────────────────────────────────
close(_io_log)
@printf("\nRésultats sauvegardés dans : %s\n", RESULTS_FILE)
