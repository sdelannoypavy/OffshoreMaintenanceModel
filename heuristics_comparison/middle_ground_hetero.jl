module HeteroMaintenance

using Random, Statistics, Printf
using JuMP, HiGHS
using Base: @kwdef

# States: 1 = good, 2 = degraded, 3 = breakdown (panne)
# π is window cost (shared); all other cost/transition params are per-component vectors.
@kwdef struct HeteroMaintenanceParams
    C::Int
    H::Int
    K::Int
    p_deg::Vector{Float64}    # prob of degradation per period  1 → 2  (length C)
    p_fail::Vector{Float64}   # prob of failure per period      2 → 3  (length C)
    p_repair::Vector{Float64} # prob of successful repair             (length C)
    h_deg::Vector{Float64}    # holding cost in state 2 (degraded)   (length C)
    h_loss::Vector{Float64}   # holding cost in state 3 (panne)      (length C)
    c_repair::Vector{Float64} # repair cost                          (length C)
    π::Float64                # window opening cost (shared)
end

# ============================================================================
# SURROGATE VALUES & GAINS — per-component (H1)
# ============================================================================

"""
u[i, s, m] = expected future cost for component i in state s with m steps
             remaining, under the PAID-REPAIR baseline (Option B).

At each step the component can either:
  - wait : incur holding cost ℓ(s) and transition naturally
  - repair: pay π + c_repair[i] (no holding cost that period) and repair
            stochastically (succeed → state 1 with prob p_repair; fail → natural
            transition with prob 1 − p_repair).

This is tighter than the no-repair baseline because it already accounts for the
possibility of standalone paid repairs in future periods.
"""
function surrogate_values(p::HeteroMaintenanceParams)
    u    = zeros(p.C, 3, p.H + 1)
    h    = [zeros(p.C) for _ in 1:3]   # holding costs per state
    h[2] = p.h_deg
    h[3] = p.h_loss

    for i in 1:p.C
        π_i = p.π + p.c_repair[i]      # full standalone repair cost
        for m in 2:p.H+1
            # expected continuation values under natural (wait) transition
            E_wait = [
                (1 - p.p_deg[i])  * u[i, 1, m-1] + p.p_deg[i]  * u[i, 2, m-1],
                (1 - p.p_fail[i]) * u[i, 2, m-1] + p.p_fail[i] * u[i, 3, m-1],
                u[i, 3, m-1]   # state 3 absorbing under no-repair
            ]
            # expected continuation values after (stochastic) repair
            E_rep = [p.p_repair[i] * u[i, 1, m-1] + (1 - p.p_repair[i]) * E_wait[s]
                     for s in 1:3]

            for s in 1:3
                wait_cost   = h[s][i] + E_wait[s]
                repair_cost = π_i     + E_rep[s]
                u[i, s, m]  = min(wait_cost, repair_cost)
            end
        end
    end
    return u
end

"""
Gnet[i, s, m] = net gain from repairing component i in state s with m steps left,
                using the shared window (no extra setup cost).

Formula (eq. 3 of companion note):
  G[i,s,m] = ℓ(s) + E[u_{m-1}(X^wait) | s] − E[u_{m-1}(X^rep) | s]

where E[u^rep | s] accounts for stochastic repair (succeed → state 1 with
prob p_repair; fail → natural transition with prob 1 − p_repair).
"""
function repair_gains(p::HeteroMaintenanceParams, u::Array{Float64,3})
    G    = zeros(p.C, 3, p.H + 1)
    Gnet = zeros(p.C, 3, p.H + 1)
    h    = (i, s) -> s == 2 ? p.h_deg[i] : (s == 3 ? p.h_loss[i] : 0.0)

    for i in 1:p.C, m in 2:p.H+1
        # expected continuation under natural (wait) transition from each state
        E_wait = [
            (1 - p.p_deg[i])  * u[i, 1, m-1] + p.p_deg[i]  * u[i, 2, m-1],
            (1 - p.p_fail[i]) * u[i, 2, m-1] + p.p_fail[i] * u[i, 3, m-1],
            u[i, 3, m-1]
        ]
        # expected continuation after (stochastic) repair
        E_rep = [p.p_repair[i] * u[i, 1, m-1] + (1 - p.p_repair[i]) * E_wait[s]
                 for s in 1:3]

        for s in 1:3
            G[i, s, m]    = h(i, s) + E_wait[s] - E_rep[s]
            Gnet[i, s, m] = G[i, s, m] - p.c_repair[i]
        end
    end
    return G, Gnet
end

function lambda_adjust(λstar, m, k, p::HeteroMaintenanceParams)
    k >= m ? 0.0 : (k == 0 ? Inf : λstar * (p.K / p.H) / (k / m))
end

# ============================================================================
# HEURISTIC 1: NET GAINS + SHADOW PRICE
# ============================================================================

function simulate_openings(p::HeteroMaintenanceParams, Gnet::Array{Float64,3},
                            λ::Float64; nsim::Int=500)
    total_openings = 0
    for _ in 1:nsim
        x = fill(1, p.C)
        opens = 0
        for m in p.H:-1:1
            Sm = sum(max(Gnet[i, x[i], m], 0.0) for i in 1:p.C)
            if Sm > λ
                opens += 1
                for i in 1:p.C
                    if Gnet[i, x[i], m] > 0
                        x[i] = 1
                    end
                end
            end
            # 3-state natural transitions
            for i in 1:p.C
                if x[i] == 1 && rand() < p.p_deg[i]
                    x[i] = 2
                elseif x[i] == 2 && rand() < p.p_fail[i]
                    x[i] = 3
                end
                # state 3 is absorbing without repair
            end
        end
        total_openings += opens
    end
    return total_openings / nsim
end

function find_lambda_star(p::HeteroMaintenanceParams, Gnet::Array{Float64,3};
                           tol=1e-3, maxit=40)
    λmin = 0.0
    λmax = p.H * sum(p.h_loss[i] + p.h_deg[i] for i in 1:p.C)
    for _ in 1:maxit
        λ = (λmin + λmax) / 2
        openings = simulate_openings(p, Gnet, λ; nsim=500)
        g = openings - p.K
        abs(g) < tol && break
        g > 0 ? (λmin = λ) : (λmax = λ)
    end
    return (λmin + λmax) / 2
end

function simulate_cycle(p::HeteroMaintenanceParams, Gnet::Array{Float64,3},
                         λstar::Float64; seed=1234)
    Random.seed!(seed)
    k = p.K
    total_loss = 0.0
    total_cost = 0.0
    x = fill(1, p.C)

    for m in p.H:-1:1
        Sm  = sum(max(Gnet[i, x[i], m], 0.0) for i in 1:p.C)
        λmk = lambda_adjust(λstar, m, k, p)
        η   = k >= 1 ? min(λmk, p.π) : p.π

        # holding cost at the START of the period (before action)
        total_loss += sum(p.h_deg[i]  * (x[i] == 2) +
                          p.h_loss[i] * (x[i] == 3) for i in 1:p.C)

        R = Int[]
        if Sm > η
            R = [i for i in 1:p.C if Gnet[i, x[i], m] > 0]
            total_cost += sum(p.c_repair[i] for i in R)
            if k >= 1 && λmk <= p.π
                k -= 1
            else
                total_cost += p.π
            end
            for i in R
                if rand() < p.p_repair[i]   # stochastic repair → state 1
                    x[i] = 1
                end
            end
        end

        # natural 3-state transitions
        for i in 1:p.C
            if x[i] == 1 && rand() < p.p_deg[i]
                x[i] = 2
            elseif x[i] == 2 && rand() < p.p_fail[i]
                x[i] = 3
            end
        end
    end

    return total_loss + total_cost
end

# ============================================================================
# SHARED UTILITY: per-component transition matrices
# ============================================================================

"""
Trans[i, s′, s, a] = P(next state = s′ | current state = s, action a).

No repair (a=1): 1→{1,2}, 2→{2,3}, 3→{3} (absorbing).
Repair   (a=2): with prob p_repair → state 1; with prob (1−p_repair) → normal transition.
"""
function build_trans(p::HeteroMaintenanceParams)
    Trans = zeros(p.C, 3, 3, 2)
    for i in 1:p.C
        # ── no repair (a=1) ──────────────────────────────────────────────────
        Trans[i, 1, 1, 1] = 1 - p.p_deg[i];  Trans[i, 2, 1, 1] = p.p_deg[i]
        Trans[i, 2, 2, 1] = 1 - p.p_fail[i]; Trans[i, 3, 2, 1] = p.p_fail[i]
        Trans[i, 3, 3, 1] = 1.0

        # ── repair (a=2): succeed → state 1; fail → normal transition ────────
        for s in 1:3, sp in 1:3
            Trans[i, sp, s, 2] = p.p_repair[i] * (sp == 1 ? 1.0 : 0.0) +
                                  (1 - p.p_repair[i]) * Trans[i, sp, s, 1]
        end
    end
    return Trans
end

# ============================================================================
# HEURISTIC 2: ROLLING HORIZON LP (soft budget in expectation)
# ============================================================================

function fluid_lp_rolling_horizon(p::HeteroMaintenanceParams, x_current::Vector{Int}, m::Int)
    if m <= 0
        return nothing, nothing, 0.0
    end

    Trans = build_trans(p)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[1:p.C, 1:3, 1:2, 0:1, 1:m] >= 0)
    @variable(model, 0 <= w_t[1:m] <= 1)
    @variable(model, y >= 0)

    # ── INITIAL CONDITION ─────────────────────────────────────────────────────
    for i in 1:p.C
        s_i = x_current[i]
        @constraint(model, sum(x[i, s_i, a, ww, m] for a in 1:2, ww in 0:1) == 1.0)
        for s in 1:3
            s != s_i && @constraint(model,
                sum(x[i, s, a, ww, m] for a in 1:2, ww in 0:1) == 0.0)
        end
    end

    # ── REPAIR REQUIRES WINDOW ────────────────────────────────────────────────
    for t in 1:m, i in 1:p.C, s in 1:3
        @constraint(model, x[i, s, 2, 0, t] == 0.0)
    end

    # ── KOLMOGOROV FLOW ───────────────────────────────────────────────────────
    for t in 1:m-1, i in 1:p.C, sp in 1:3
        flow_to   = sum(x[i, sp, a, ww, t] for a in 1:2, ww in 0:1)
        flow_from = sum(Trans[i, sp, s, a] * x[i, s, a, ww, t+1]
                        for s in 1:3, a in 1:2, ww in 0:1)
        @constraint(model, flow_to == flow_from)
    end

    # ── WINDOW SHARING ────────────────────────────────────────────────────────
    # Equality: the shared window probability must be the same for all components.
    # ww=1 covers any action (repair or wait) in an open window period.
    for t in 1:m, i in 1:p.C
        @constraint(model, w_t[t] == sum(x[i, s, a, 1, t] for s in 1:3, a in 1:2))
    end

    # ── NORMALISATION ─────────────────────────────────────────────────────────
    for t in 1:m, i in 1:p.C
        @constraint(model,
            sum(x[i, s, a, ww, t] for s in 1:3, a in 1:2, ww in 0:1) == 1.0)
    end

    # ── BUDGET (soft, penalised) ───────────────────────────────────────────────
    @constraint(model, y >= sum(w_t[t] for t in 1:m) - p.K)

    # ── OBJECTIVE ─────────────────────────────────────────────────────────────
    repair_cost  = sum(p.c_repair[i] * x[i, s, 2, ww, t]
                       for i in 1:p.C, s in 1:3, ww in 0:1, t in 1:m)
    holding_cost = sum(p.h_deg[i]  * x[i, 2, a, ww, t] +
                       p.h_loss[i] * x[i, 3, a, ww, t]
                       for i in 1:p.C, a in 1:2, ww in 0:1, t in 1:m)
    window_cost  = p.π * y

    @objective(model, Min, repair_cost + holding_cost + window_cost)

    optimize!(model)

    status = termination_status(model)
    if status != MOI.OPTIMAL && status != MOI.FEASIBLE_POINT
        @warn "LP(m=$m) failed: $status"
        return nothing, nothing, NaN
    end

    return value.(x), value.(w_t), objective_value(model)
end

function simulate_cycle_lp_marginal(p::HeteroMaintenanceParams; seed=1234)
    Random.seed!(seed)
    k = p.K
    total_loss = 0.0
    total_cost = 0.0
    x = fill(1, p.C)

    for m in p.H:-1:1
        x_opt, w_opt, _ = fluid_lp_rolling_horizon(p, x, m)

        R   = Int[]
        w_m = 0.0

        if x_opt !== nothing
            for i in 1:p.C
                s_i = x[i]
                if sum(x_opt[i, s_i, 2, ww, m] for ww in 0:1) > 0.5
                    push!(R, i)
                end
            end
            w_m = w_opt[m]
        end

        # holding cost at the START of the period (before action)
        total_loss += sum(p.h_deg[i]  * (x[i] == 2) +
                          p.h_loss[i] * (x[i] == 3) for i in 1:p.C)

        if !isempty(R)
            total_cost += sum(p.c_repair[i] for i in R)
            if w_m > 0.5 && k >= 1
                k -= 1
            else
                total_cost += p.π
            end
            for i in R
                if rand() < p.p_repair[i]
                    x[i] = 1
                end
            end
        end

        for i in 1:p.C
            if x[i] == 1 && rand() < p.p_deg[i]
                x[i] = 2
            elseif x[i] == 2 && rand() < p.p_fail[i]
                x[i] = 3
            end
        end
    end

    return total_loss + total_cost
end

# ============================================================================
# HEURISTIC 3: ROLLING HORIZON LP — k in state, hard window constraint
# ============================================================================

"""
State per component: (s, k) — condition s ∈ {1,2,3}, remaining free windows k ∈ {0,...,K}
Action:             (a, w) — repair a ∈ {1,2}, window type w ∈ {0=paid/none, 1=free}

The free window is a SHARED event: if it is open at period t, k decreases for
ALL components regardless of whether they repair (so (a=1, w=1) is valid).
Only forbidden: w=1 when k=0.
"""
function fluid_lp_deterministic_windows(p::HeteroMaintenanceParams,
                                         x_current::Vector{Int},
                                         k_current::Int, m::Int)
    if m <= 0
        return nothing, nothing, nothing, 0.0
    end

    K     = p.K
    Trans = build_trans(p)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[1:p.C, 1:3, 0:K, 1:2, 0:1, 1:m] >= 0)
    @variable(model, 0 <= w_t_free[0:K, 1:m] <= 1)
    @variable(model, 0 <= w_t_paid[0:K, 1:m] <= 1)

    # ── INITIAL CONDITION ─────────────────────────────────────────────────────
    for i in 1:p.C
        s_i = x_current[i]
        @constraint(model,
            sum(x[i, s_i, k_current, a, w, m] for a in 1:2, w in 0:1) == 1.0)
        for s in 1:3, k in 0:K
            if s != s_i || k != k_current
                @constraint(model,
                    sum(x[i, s, k, a, w, m] for a in 1:2, w in 0:1) == 0.0)
            end
        end
    end

    # ── INVALID ACTIONS: w=1 forbidden when k=0 ──────────────────────────────
    for t in 1:m, i in 1:p.C, s in 1:3
        @constraint(model, x[i, s, 0, 2, 1, t] == 0.0)
        @constraint(model, x[i, s, 0, 1, 1, t] == 0.0)
    end

    # ── KOLMOGOROV FLOW ───────────────────────────────────────────────────────
    # w=0: k unchanged; w=1: k decreases by 1 (kp+1 → kp)
    for t in 1:m-1, i in 1:p.C, sp in 1:3, kp in 0:K
        flow_to   = sum(x[i, sp, kp, a, w, t] for a in 1:2, w in 0:1)
        flow_from = sum(Trans[i, sp, s, a] * x[i, s, kp, a, 0, t+1]
                        for s in 1:3, a in 1:2)
        if kp + 1 <= K
            flow_from += sum(Trans[i, sp, s, a] * x[i, s, kp+1, a, 1, t+1]
                             for s in 1:3, a in 1:2)
        end
        @constraint(model, flow_to == flow_from)
    end

    # ── WINDOW SHARING ────────────────────────────────────────────────────────
    # Equality: the shared window probability must be the same for all components,
    # conditioning on each budget level k separately.
    for t in 1:m, i in 1:p.C, k in 0:K
        @constraint(model,
            w_t_free[k, t] == sum(x[i, s, k, a, 1, t] for s in 1:3, a in 1:2))
        @constraint(model,
            w_t_paid[k, t] == sum(x[i, s, k, 2, 0, t] for s in 1:3))
    end

    # ── NORMALISATION ─────────────────────────────────────────────────────────
    for t in 1:m, i in 1:p.C
        @constraint(model,
            sum(x[i, s, k, a, w, t]
                for s in 1:3, k in 0:K, a in 1:2, w in 0:1) == 1.0)
    end

    # ── OBJECTIVE ─────────────────────────────────────────────────────────────
    repair_cost  = sum(p.c_repair[i] * x[i, s, k, 2, w, t]
                       for i in 1:p.C, s in 1:3, k in 0:K, w in 0:1, t in 1:m)
    holding_cost = sum(p.h_deg[i]  * x[i, 2, k, a, w, t] +
                       p.h_loss[i] * x[i, 3, k, a, w, t]
                       for i in 1:p.C, k in 0:K, a in 1:2, w in 0:1, t in 1:m)
    window_cost  = p.π * sum(w_t_paid[k, t] for k in 0:K, t in 1:m)

    @objective(model, Min, repair_cost + holding_cost + window_cost)

    optimize!(model)

    status = termination_status(model)
    if status != MOI.OPTIMAL && status != MOI.FEASIBLE_POINT
        @warn "LP_det(m=$m, k=$k_current) failed: $status"
        return nothing, nothing, nothing, NaN
    end

    return value.(x), value.(w_t_free), value.(w_t_paid), objective_value(model)
end

function simulate_cycle_lp_deterministic(p::HeteroMaintenanceParams; seed=1234)
    Random.seed!(seed)
    k = p.K
    total_loss = 0.0
    total_cost = 0.0
    x = fill(1, p.C)

    for m in p.H:-1:1
        x_opt, w_free_opt, _, _ = fluid_lp_deterministic_windows(p, x, k, m)

        R        = Int[]
        use_free = false

        if x_opt !== nothing
            for i in 1:p.C
                s_i = x[i]
                if x_opt[i, s_i, k, 2, 1, m] + x_opt[i, s_i, k, 2, 0, m] > 0.5
                    push!(R, i)
                end
            end
            use_free = k > 0 && w_free_opt[k, m] >= 0.5
        end

        # holding cost at the START of the period (before action)
        total_loss += sum(p.h_deg[i]  * (x[i] == 2) +
                          p.h_loss[i] * (x[i] == 3) for i in 1:p.C)

        if !isempty(R)
            total_cost += sum(p.c_repair[i] for i in R)
            if use_free
                k -= 1
            else
                total_cost += p.π
            end
            for i in R
                if rand() < p.p_repair[i]
                    x[i] = 1
                end
            end
        end

        for i in 1:p.C
            if x[i] == 1 && rand() < p.p_deg[i]
                x[i] = 2
            elseif x[i] == 2 && rand() < p.p_fail[i]
                x[i] = 3
            end
        end
    end

    return total_loss + total_cost
end

# ============================================================================
# LP LOWER BOUNDS (full horizon, all-healthy start, k = K)
# ============================================================================

function lp_lower_bound_rolling(p::HeteroMaintenanceParams)
    _, _, lb = fluid_lp_rolling_horizon(p, fill(1, p.C), p.H)
    return lb
end

function lp_lower_bound_deterministic(p::HeteroMaintenanceParams)
    _, _, _, lb = fluid_lp_deterministic_windows(p, fill(1, p.C), p.K, p.H)
    return lb
end

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

function run_experiment_hetero(; seed_params=42, nsim=100)
    println("\n" * "="^80)
    println("HETEROGENEOUS MAINTENANCE — 3 STATES (good / degraded / panne)")
    println("="^80)

    rng = Random.MersenneTwister(seed_params)
    C, H, K = 20, 24, 3

    p = HeteroMaintenanceParams(
        C        = C,
        H        = H,
        K        = K,
        p_deg    = rand(rng, C) .* 0.20 .+ 0.05,   # U[0.05, 0.25]  1→2
        p_fail   = rand(rng, C) .* 0.30 .+ 0.05,   # U[0.05, 0.35]  2→3
        p_repair = fill(1.0, C),                    # perfect repair for now
        h_deg    = rand(rng, C) .* 0.50 .+ 0.10,   # U[0.10, 0.60]
        h_loss   = rand(rng, C) .* 2.00 .+ 0.50,   # U[0.50, 2.50]
        c_repair = rand(rng, C) .* 0.30 .+ 0.05,   # U[0.05, 0.35]
        π        = 5.0
    )

    println("\nGlobal: C=$C  H=$H  K=$K  π=$(p.π)")
    println("\n  Comp │ p_deg │ p_fail │ p_repair │ h_deg │ h_loss │ c_repair")
    println("  ─────┼───────┼────────┼──────────┼───────┼────────┼─────────")
    for i in 1:C
        @printf("    %d  │ %.3f │  %.3f  │  %.3f   │ %.3f │  %.3f  │  %.3f\n",
                i, p.p_deg[i], p.p_fail[i], p.p_repair[i],
                p.h_deg[i], p.h_loss[i], p.c_repair[i])
    end

    # ── LP LOWER BOUNDS ───────────────────────────────────────────────────────
    println("\n" * "-"^80)
    println("LP LOWER BOUNDS (full horizon, all-healthy start)")
    println("-"^80)
    lb_rolling = lp_lower_bound_rolling(p)
    println("  LP rolling  (soft budget)    : ", round(lb_rolling, digits=4))
    lb_det = lp_lower_bound_deterministic(p)
    println("  LP det.     (k in state)     : ", round(lb_det,     digits=4))

    # ── HEURISTIC 1 ───────────────────────────────────────────────────────────
    println("\n" * "-"^80)
    println("HEURISTIC 1: NET GAINS + SHADOW PRICE")
    println("-"^80)
    u = surrogate_values(p)
    _, Gnet = repair_gains(p, u)
    λstar = find_lambda_star(p, Gnet)
    println("λ* ≈ ", round(λstar, digits=3))
    costs_h1 = [simulate_cycle(p, Gnet, λstar; seed=s) for s in 1:nsim]
    println("Results ($nsim sims):  mean=$(round(mean(costs_h1),digits=4))  std=$(round(std(costs_h1),digits=4))")

    # ── HEURISTIC 3 ───────────────────────────────────────────────────────────
    println("\n" * "-"^80)
    println("HEURISTIC 3: ROLLING LP (k in state, hard window constraint)")
    println("-"^80)
    costs_h3 = [simulate_cycle_lp_deterministic(p; seed=s) for s in 1:nsim]
    println("Results ($nsim sims):  mean=$(round(mean(costs_h3),digits=4))  std=$(round(std(costs_h3),digits=4))")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    println("\n" * "="^80)
    println("SUMMARY")
    println("="^80)
    @printf("  %-28s  %8s  %10s\n", "Method", "Mean", "Gap vs LB1")
    println("  " * "─"^52)
    @printf("  %-28s  %8.4f\n", "LP bound (rolling)",      lb_rolling)
    @printf("  %-28s  %8.4f\n", "LP bound (det. windows)", lb_det)
    for (label, costs) in [("H1 (gains + shadow price)", costs_h1),
                            ("H3 (det. windows)        ", costs_h3)]
        gap = (mean(costs) / lb_rolling - 1) * 100
        @printf("  %-28s  %8.4f  %+6.1f%%\n", label, mean(costs), gap)
    end
    println("="^80)
end

run_experiment_hetero()

end # module HeteroMaintenance
