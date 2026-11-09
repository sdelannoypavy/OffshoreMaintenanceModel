# direct_lp.jl
#
# Alternative to column generation: a monolithic LP that embeds all pricing
# variables directly into the master LP for every time step, with integrality
# relaxed to [0,1].
#
# q[c,x,m,(k,κ),t] : probability mass for component c in state x, mode m,
#                     with k remaining quota, using κ free days in period t.
#                     Only valid pairs (κ ≤ k ≤ K0) are created.
#
# z[c,t]    ∈ [0,1] : probability of maintaining component c at step t
# free[τ,t] ∈ [0,1] : probability of declaring day τ of period t a free day
#
# Coupling constraints:
#   (m)  Σ_{x,k,κ} q[c,x,2,(k,κ),t] ≤ z[c,t]          maintenance action
#   (κ)  Σ_{x,m,(k,κ)} κ·q[1,x,m,(k,κ),t] = Σ_τ free[τ,t]  free-day action
#
# Quota dynamics are embedded in the flow constraint (no global quota constraint).
# Action at t=1 is extracted by rounding (threshold 0.5 on z and free).
#
# Requires column_generation.jl to be loaded (provides MaintenanceColumn,
# MaintenancePlan, optimizer_cg, stage_tcyc, stage_year).

using Printf

# ─────────────────────────────────────────────────────────────────────────────
# LP builder + solver
# ─────────────────────────────────────────────────────────────────────────────

function solve_direct_LP(x0::Vector{Int}, k0::Int, K0::Int, t0::Int,
                          h_optim, pr,
                          n::Vector{Int}, d::Vector{Int},
                          C::Int, comp_cost::Vector{Float64}, n_max::Int,
                          δ_simple, p_simple;
                          H_full::Int=6,
                          r::Vector{Int}=ones(Int, C),
                          R::Int=3,
                          S_price::Int=20,
                          fixed_start_day::Union{Int,Nothing}=15,
                          margin::Int=10,
                          optimizer=optimizer_cg)

    t0_cyc      = (t0 - 1) % 6 + 1
    nb_scen_all = size(h_optim, 2)
    S           = min(S_price, nb_scen_all)

    # ── Per-period metadata ────────────────────────────────────────────────────
    t_cyc_vec = [stage_tcyc(t, t0_cyc) for t in 1:H_full]
    T_per_vec = [62 - count(==(-1), h_optim[t_cyc_vec[t], 1, :]) for t in 1:H_full]
    T_end_vec = [max(T_per_vec[t] - margin, 1) for t in 1:H_full]
    T_per_max = maximum(T_per_vec)

    # year_resets[j] = true if a quota reset occurs at t=j+1 (new calendar year)
    year_resets = [stage_year(t+1, t0_cyc) > stage_year(t, t0_cyc) for t in 1:H_full-1]

    # Valid (k, κ) pairs: 0 ≤ κ ≤ k ≤ K0
    valid_kκ = [(k, κ) for k in 0:K0 for κ in 0:k]

    # Precompute weather and price matrices (constants in the LP)
    h_mat  = zeros(Int,     H_full, S,     T_per_max)
    pr_mat = zeros(Float64, H_full,        T_per_max)
    for t in 1:H_full
        tc = t_cyc_vec[t]; Tp = T_per_vec[t]
        for s in 1:S
            h_mat[t, s, 1:Tp] = h_optim[tc, s, 1:Tp]
        end
        pr_mat[t, 1:Tp] = pr[tc, 1:Tp]
    end

    model = Model(optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "Method", 1)

    # ── State-action variables ─────────────────────────────────────────────────
    # q[c,x,m,(k,κ),t] : only valid pairs (κ ≤ k) are created.
    @variable(model, q[c=1:C, x=1:n[c], m=1:2, (k,κ)=valid_kκ, t=1:H_full] >= 0)

    # Fix q = 0 for k ≠ k0 at t=1 (initial remaining quota is k0).
    for c in 1:C, x in 1:n[c], m in 1:2, (k, κ) in valid_kκ
        k != k0 && fix(q[c, x, m, (k, κ), 1], 0.0; force=true)
    end

    # ── Pricing variables (continuous relaxation) ──────────────────────────────
    @variable(model, 0 <= z[1:C,           1:H_full              ] <= 1)
    @variable(model, 0 <= m_start[1:C,     1:H_full, 1:T_per_max ] <= 1)
    @variable(model, 0 <= ong_m[1:C, 1:S,  1:T_per_max, 1:H_full ] <= 1)
    @variable(model, q_var[1:C,      1:S,  1:T_per_max, 1:H_full ] >= 0)
    @variable(model, 0 <= any_maint[  1:S, 1:T_per_max, 1:H_full ] <= 1)
    @variable(model, 0 <= free[        1:T_per_max, 1:H_full      ] <= 1)
    @variable(model, 0 <= w[      1:S, 1:T_per_max, 1:H_full      ] <= 1)

    # ── Fix pricing variables outside each period's valid range ────────────────
    for t in 1:H_full
        Tp = T_per_vec[t]; Te = T_end_vec[t]
        for τ in Tp+1:T_per_max
            for c in 1:C
                fix(m_start[c, t, τ], 0.0; force=true)
                for s in 1:S
                    fix(ong_m[c, s, τ, t], 0.0; force=true)
                    fix(q_var[c, s, τ, t], 0.0; force=true)
                end
            end
            for s in 1:S
                fix(any_maint[s, τ, t], 0.0; force=true)
                fix(w[s, τ, t], 0.0; force=true)
            end
            fix(free[τ, t], 0.0; force=true)
        end
        for τ in Te+1:Tp
            for c in 1:C; fix(m_start[c, t, τ], 0.0; force=true); end
        end
        if !isnothing(fixed_start_day)
            for τ in 1:Te
                τ == fixed_start_day && continue
                for c in 1:C; fix(m_start[c, t, τ], 0.0; force=true); end
            end
        end
    end

    # ── Initial state (x = x0, k = k0) ────────────────────────────────────────
    α = zeros(C, n_max)
    for c in 1:C; α[c, x0[c]] = 1.0; end
    @constraint(model, init[c=1:C, x=1:n[c]],
        sum(q[c, x, m, (k0, κ), 1] for m in 1:2, κ in 0:k0) == α[c, x])

    # ── Flow: Markov dynamics on x + quota dynamics on k ──────────────────────
    # No-reset : k_new = k_prev - κ_prev  →  k_prev = k_new + κ_prev
    # Year reset: k_new = K0 for any (k_prev, κ_prev)
    for t in 2:H_full, c in 1:C, x in 1:n[c], k_new in 0:K0
        tc_prev = stage_tcyc(t-1, t0_cyc)
        resets  = year_resets[t-1]

        lhs = sum(q[c, x, m, (k_new, κ), t] for m in 1:2, κ in 0:k_new)

        if resets
            k_new != K0 && continue
            rhs = sum(p_simple[c, xp, x, m, tc_prev] * q[c, xp, m, (k, κ), t-1]
                      for xp in 1:n[c], m in 1:2, (k, κ) in valid_kκ)
        else
            rhs = sum(p_simple[c, xp, x, m, tc_prev] *
                      q[c, xp, m, (k_new + κ_prev, κ_prev), t-1]
                      for xp in 1:n[c], m in 1:2, κ_prev in 0:K0-k_new)
        end

        @constraint(model, lhs >= rhs)
    end

    # ── Coupling: q ↔ pricing variables ───────────────────────────────────────
    # (m) maintenance action: total maintenance probability bounded by z[c,t]
    @constraint(model, coupling_m[c=1:C, t=1:H_full],
        sum(q[c, x, 2, (k, κ), t] for x in 1:n[c], (k, κ) in valid_kκ) <= z[c, t])

    # (κ) free-days action: expected free days from q equals sum of free[τ,t].
    # Uses component c=1 as reference (κ is a global action shared by all c).
    @constraint(model, coupling_κ[t=1:H_full],
        sum(κ * q[1, x, m, (k, κ), t]
            for x in 1:n[1], m in 1:2, (k, κ) in valid_kκ)
        == sum(free[τ, t] for τ in 1:T_per_vec[t]))

    # ── Per-period pricing constraints (relaxed) ───────────────────────────────
    @constraint(model, [c=1:C, t=1:H_full],
        sum(m_start[c, t, τ] for τ in 1:T_end_vec[t]) == z[c, t])

    for t in 1:H_full
        Tp = T_per_vec[t]; Te = T_end_vec[t]
        for c in 1:C, s in 1:S
            @constraint(model, q_var[c, s, 1, t] == d[c] * m_start[c, t, 1])
            for τ in 1:Tp-1
                if τ + 1 <= Te
                    @constraint(model,
                        q_var[c, s, τ+1, t] == q_var[c, s, τ, t]
                            - h_mat[t, s, τ] * ong_m[c, s, τ, t]
                            + d[c] * m_start[c, t, τ+1])
                else
                    @constraint(model,
                        q_var[c, s, τ+1, t] == q_var[c, s, τ, t]
                            - h_mat[t, s, τ] * ong_m[c, s, τ, t])
                end
            end
        end
    end

    @constraint(model, [c=1:C, s=1:S, τ=1:T_per_max, t=1:H_full],
        ong_m[c, s, τ, t] <= q_var[c, s, τ, t])
    @constraint(model, [c=1:C, s=1:S, τ=1:T_per_max, t=1:H_full],
        ong_m[c, s, τ, t] >= (1.0 / d[c]) * q_var[c, s, τ, t])
    @constraint(model, [s=1:S, t=1:H_full, τ=1:T_per_vec[t]],
        sum(r[c] * ong_m[c, s, τ, t] for c in 1:C) <= R)
    @constraint(model, [c=1:C, s=1:S, t=1:H_full, τ=1:T_per_vec[t]],
        any_maint[s, τ, t] >= ong_m[c, s, τ, t])
    @constraint(model, [s=1:S, t=1:H_full, τ=1:T_per_vec[t]], w[s, τ, t] <= free[τ, t])
    @constraint(model, [s=1:S, t=1:H_full, τ=1:T_per_vec[t]], w[s, τ, t] <= any_maint[s, τ, t])
    @constraint(model, [s=1:S, t=1:H_full, τ=1:T_per_vec[t]],
        w[s, τ, t] >= free[τ, t] + any_maint[s, τ, t] - 1)

    # ── Objective ──────────────────────────────────────────────────────────────
    @objective(model, Min,
        sum(δ_simple[c, x, stage_tcyc(t, t0_cyc)] * q[c, x, 1, (k, κ), t]
            for c in 1:C, x in 1:n[c], (k, κ) in valid_kκ, t in 1:H_full)
        + sum(comp_cost[c] * z[c, t] for c in 1:C, t in 1:H_full)
        + sum((1.0 / S) * pr_mat[t, τ] * (any_maint[s, τ, t] - w[s, τ, t])
              for t in 1:H_full, s in 1:S, τ in 1:T_per_vec[t]))

    optimize!(model)

    status = termination_status(model)
    obj    = status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) ? objective_value(model) : Inf
    @printf("  [direct_lp] Status: %s  Obj: %.6f  (%d vars, S=%d)\n",
            status, obj, num_variables(model), S)

    return model, status
end

# ─────────────────────────────────────────────────────────────────────────────
# Policy extraction: round LP solution at t=1 to a deterministic action
# ─────────────────────────────────────────────────────────────────────────────

function extract_policy_direct(model, C::Int, T_per_1::Int,
                                fixed_start_day::Union{Int,Nothing})
    z_val        = value.(model[:z])
    m_start_val  = value.(model[:m_start])
    free_val     = value.(model[:free])

    comps_found      = Int[]
    start_days_found = Int[]
    for c in 1:C
        z_val[c, 1] > 0.5 || continue
        push!(comps_found, c)
        if !isnothing(fixed_start_day)
            push!(start_days_found, fixed_start_day)
        else
            best_τ = argmax(m_start_val[c, 1, 1:T_per_1])
            push!(start_days_found, best_τ)
        end
    end

    free_days = [τ for τ in 1:T_per_1 if free_val[τ, 1] > 0.5]

    col_star = MaintenanceColumn(MaintenancePlan(comps_found, start_days_found), free_days)
    m_opt    = zeros(Int, C)
    for c in comps_found; m_opt[c] = 1; end

    return m_opt, col_star, free_days
end

# ─────────────────────────────────────────────────────────────────────────────
# Cache and interface (same key structure as π_cg)
# ─────────────────────────────────────────────────────────────────────────────

const cache_direct_lp = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64},
                              Tuple{Vector{Int64}, MaintenanceColumn, Vector{Int64}}}()

function load_cache_direct_lp!(file::String)
    !isfile(file) && return
    local saved
    @load file saved
    merge!(cache_direct_lp, saved)
    @printf("  Direct LP cache loaded: %d policies from %s\n",
            length(cache_direct_lp), file)
end

function save_cache_direct_lp(file::String)
    saved = cache_direct_lp
    @save file saved
end

function π_direct_lp(x0, k0, K0, t0,
                      h_optim, pr, n, d, C,
                      comp_cost, n_max, δ_simple, p_simple;
                      H_full=6, r=ones(Int, C), R=3,
                      S_price=20,
                      fixed_start_day::Union{Int,Nothing}=15,
                      cache_file::Union{String,Nothing}=nothing)

    cle = (Tuple(x0), k0, t0)

    if !haskey(cache_direct_lp, cle)
        t0_cyc  = (t0 - 1) % 6 + 1
        T_per_1 = 62 - count(==(-1), h_optim[t0_cyc, 1, :])

        model, status = solve_direct_LP(x0, k0, K0, t0,
            h_optim, pr, n, d, C, comp_cost, n_max, δ_simple, p_simple;
            H_full=H_full, r=r, R=R, S_price=S_price,
            fixed_start_day=fixed_start_day)

        if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
            m_opt, col_star, free_days =
                extract_policy_direct(model, C, T_per_1, fixed_start_day)
        else
            @printf("  [direct_lp] Infeasible or error — returning no-action.\n")
            col_star  = MaintenanceColumn(MaintenancePlan(Int[], Int[]), Int[])
            m_opt     = zeros(Int, C)
            free_days = Int[]
        end

        cache_direct_lp[cle] = (m_opt, col_star, free_days)
        !isnothing(cache_file) && save_cache_direct_lp(cache_file)
    end

    return cache_direct_lp[cle]
end
