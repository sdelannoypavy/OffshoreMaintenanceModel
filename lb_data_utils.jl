# lb_data_utils.jl
#
# Fonctions partagées entre lower_bound.jl et script_simple_pole.jl :
#   - create_δ_full  : calcul de δ (coûts panne/capacité, sans comp_cost)
#   - create_p_full  : calcul de p (transitions, m binaire : 1=non maintenu, 2=maintenu)
#   - _need_recompute_δ / _need_recompute_p : détection de fichiers invalides

# ── create_δ : nb_time fenêtres, sans coûts fixes de maintenance ─────────────
function create_δ_full(C, h_optim, pr, n, d, T, P_daily, delta_file, nb_time, P_evac)
    Tmax_computeδ = T + 6
    n_max_loc     = maximum(n)
    nb_scen       = size(h_optim, 2)

    δ = zeros(C, n_max_loc, nb_time, Q + 1, Tmax_computeδ)

    for c in 1:C, x in 1:n[c], m_global in 1:nb_time, κ in 0:Q, t in 1:Tmax_computeδ
        T_per = 62 - count(==(-1), h_optim[t, 1, :])
        m     = m_global - 1   # m=0 : pas de maintenance, m≥1 : maintenance

        Policies_m_t = create_schedule_m(m > 0 ? 1 : 0, t_start, T_per)
        Policies_κ_t = create_schedule_κ(κ, t_start, T_per)

        cost = 0.0
        for scen in 1:nb_scen
            cost += (1/nb_scen) * average_period_cost(
                T_per, Policies_m_t, Policies_κ_t,
                x, h_optim[t, scen, :], pr[t, :],
                n[c], d[c], P_daily[c], P_evac[c, :])
        end
        δ[c, x, m_global, κ+1, t] = cost
    end

    tol = 1e-10
    for i in eachindex(δ); abs(δ[i]) < tol && (δ[i] = 0.0); end

    @save delta_file δ
    @printf("  δ sauvegardé : %s  shape=%s\n", delta_file, string(size(δ)))
end

# ── create_p : m binaire (1=non maintenu, 2=maintenu) ────────────────────────
function create_p_full(C, T, n, P_period, p_file, nb_time)
    Tmax_computeδ = T + 6
    n_max_loc     = maximum(n)

    # dim 6 = 2 (binaire) : m=1 pas de maintenance, m=2 maintenance
    p = zeros(C, n_max_loc, Q+1, n_max_loc, Q+1, 2, Q+1, Tmax_computeδ)

    for c in 1:C, t in 1:Tmax_computeδ, k in 0:Q, x in 1:n[c], κ in 0:k, m in 1:2
        if m == 2  # maintenance → reset à l'état 1
            if (t % 6 == 0)
                p[c, x, k+1, 1, Q+1,   m, κ+1, t] = 1.0
            else
                p[c, x, k+1, 1, k-κ+1, m, κ+1, t] = 1.0
            end
        else  # m == 1 : pas de maintenance → transitions stochastiques
            for x′ in 1:n[c]
                if (t % 6 == 0)
                    p[c, x, k+1, x′, Q+1,   m, κ+1, t] = P_period[c][x′, x]
                else
                    p[c, x, k+1, x′, k-κ+1, m, κ+1, t] = P_period[c][x′, x]
                end
            end
        end
    end

    tol = 1e-10
    for i in eachindex(p); abs(p[i]) < tol && (p[i] = 0.0); end

    @save p_file p
    @printf("  p sauvegardé : %s  shape=%s\n", p_file, string(size(p)))
end

# ── Détection de fichiers invalides ──────────────────────────────────────────
# δ attendu : (C, n_max, nb_time, Q+1, 12)
# p attendu : (C, n_max, Q+1, n_max, Q+1, 2, Q+1, 12)

function _need_recompute_δ(f, nb_time, Q)
    !isfile(f) && return true
    local δ; @load f δ
    return !(ndims(δ) == 5 && size(δ,3) == nb_time && size(δ,4) == Q+1)
end

function _need_recompute_p(f, nb_time, Q)
    !isfile(f) && return true
    local p; @load f p
    return !(ndims(p) == 8 && size(p,6) == 2 && size(p,5) == Q+1)
end
