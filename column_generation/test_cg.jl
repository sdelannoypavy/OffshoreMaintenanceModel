using JLD2, Printf

include("../get_scenarios.jl")
include("../create_scenarios_prod.jl")
include("../build_MC_utils.jl")
include("../create_simple_pole.jl")
include("../parameters_weakly_coupled.jl")
include("column_generation.jl")

# ── Constantes globales (définies dans script.jl, redéfinies ici) ─────────────
Q       = 6   # quota annuel de jours de maintenance gratuits

# ── Paramètres ────────────────────────────────────────────────────────────────
H_full = 6    # horizon = 1 an (rolling horizon)
r      = ones(Int, C)
R      = 3    # au plus 3 composants simultanément

# ── État de test ──────────────────────────────────────────────────────────────
#x0 = ones(Int, C)   # tous les composants neufs
x0 = [1, 3, 1, 1, 1, 1, 1]
k0 = Q              # quota restant pour l'année courante
K0 = Q              # quota annuel plein
t0 = 5              # stage courant (ex: 5 = nov-déc → chevauche 2 années)

@printf("État initial : x0=%s  k0=%d  t0=%d (t_cyc=%d)\n",
        string(x0), k0, t0, (t0-1)%6+1)
@printf("Knapsack : ∑r_c ≤ %d\n\n", R)

# ── Données simplifiées (précalculées une fois) ───────────────────────────────
p_simple = compute_p_simple(C, n, n_max, P_period)
δ_simple = compute_δ_simple(C, h_optim, pr, n, d, P_daily, P_evac)

# ── Génération de colonnes ────────────────────────────────────────────────────
model, B_vars, cols, obj, obj_hist = column_generation_solve(
    x0, k0, K0, t0, H_full, h_optim, pr, n, d, C,
    comp_cost, n_max, δ_simple, p_simple;
    r=r, R=R, max_iter=5, S_price=10,
    time_limit_pricing=20.0, verbose=true)

# ── Vérifications ─────────────────────────────────────────────────────────────
n_B = length(B_vars)
@printf("\nVariables B : %d au total\n", n_B)
@printf("Historique LP : %s\n", string(round.(obj_hist; digits=4)))

@printf("\nColonnes retenues (%d) :\n", length(cols))
for (i, col) in enumerate(cols)
    @printf("  [%d] comps=%s  starts=%s  free=%s\n",
            i, string(col.plan.comps), string(col.plan.start_days),
            string(col.free_days))
end

# ── Politique extraite au stage t=1 ──────────────────────────────────────────
m_opt, free_days, col_star = extract_policy_cg(cols, B_vars, C)
@printf("\nPolitique au stage t=1 :\n")
@printf("  Composants maintenus : %s\n", string(findall(==(1), m_opt)))
@printf("  Jours libres         : %s\n", string(free_days))


# pas stable sans doute car on tire des scenarios aleatoirement dans le pricing
# arrêter avant convergence de la génération de colonnes, de toute façon c'est une heuristique
# beaucoup de schémas ont quasiment le même coût, d'où longueur de convergence (par exemple décalage dans le temps... introduire une penalisation pour que ce soit moins dégénéré ?)
#