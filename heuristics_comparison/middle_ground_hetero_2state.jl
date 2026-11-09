module HeteroMaintenance2S

using Random, Statistics, Printf
using JuMP, HiGHS
using Base: @kwdef

# States: 1 = good, 2 = failed (absorbing without repair)
# π is window cost (shared); all other params are per-component vectors.
@kwdef struct HeteroParams2S
    C::Int
    H::Int
    K::Int
    p_fail::Vector{Float64}   # prob of failure per period  1 → 2  (length C)
    p_repair::Vector{Float64} # prob of successful repair            (length C)
    h::Vector{Float64}        # holding cost in state 2 (failed)    (length C)
    c_repair::Vector{Float64} # marginal repair cost                 (length C)
    π::Float64                # window opening cost (shared)
end

# ============================================================================
# SURROGATE VALUES & GAINS — per-component (H1)
# ============================================================================

"""
u[i, s, m] = expected future cost for component i in state s with m steps
             remaining, under the PAID-REPAIR baseline (Option B).

At each step the component can either:
  - wait  : incur holding cost h[i]*(s==2) and transition naturally
  - repair: pay π + c_repair[i] (no holding cost that period) and repair
            stochastically (succeed → state 1 with prob p_repair[i];
            fail → natural transition with prob 1 − p_repair[i]).
"""
function surrogate_values(p::HeteroParams2S)
    u = zeros(p.C, 2, p.H + 1)
    for i in 1:p.C
        π_i = p.π + p.c_repair[i]
        for m in 2:p.H+1
            # expected continuation under natural (wait) transition
            E_wait = [
                (1 - p.p_fail[i]) * u[i, 1, m-1] + p.p_fail[i] * u[i, 2, m-1],
                u[i, 2, m-1]   # state 2 absorbing under no-repair
            ]
            # expected continuation after stochastic repair
            E_rep = [p.p_repair[i] * u[i, 1, m-1] + (1 - p.p_repair[i]) * E_wait[s]
                     for s in 1:2]

            holding = [0.0, p.h[i]]
            for s in 1:2
                u[i, s, m] = min(holding[s] + E_wait[s],
                                 π_i        + E_rep[s])
            end
        end
    end
    return u
end

"""
G[i,s,m]    = gross gain from repairing component i in state s with m steps left.
Gnet[i,s,m] = net gain (gross − marginal repair cost c_repair[i]).

Formula: G[i,s,m] = ℓ(s) + E[u_{m-1}(X^wait)|s] − E[u_{m-1}(X^rep)|s]
"""
function repair_gains(p::HeteroParams2S, u::Array{Float64,3})
    G    = zeros(p.C, 2, p.H + 1)
    Gnet = zeros(p.C, 2, p.H + 1)
    for i in 1:p.C, m in 2:p.H+1
        E_wait = [
            (1 - p.p_fail[i]) * u[i, 1, m-1] + p.p_fail[i] * u[i, 2, m-1],
            u[i, 2, m-1]
        ]
        E_rep = [p.p_repair[i] * u[i, 1, m-1] + (1 - p.p_repair[i]) * E_wait[s]
                 for s in 1:2]
        holding = [0.0, p.h[i]]
        for s in 1:2
            G[i, s, m]    = holding[s] + E_wait[s] - E_rep[s]
            Gnet[i, s, m] = G[i, s, m] - p.c_repair[i]
        end
    end
    return G, Gnet
end

function lambda_adjust(λstar, m, k, p::HeteroParams2S)
    k >= m ? 0.0 : (k == 0 ? Inf : λstar * (p.K / p.H) / (k / m))
end

# ============================================================================
# HEURISTIC 1: NET GAINS + SHADOW PRICE
# ============================================================================

function simulate_openings(p::HeteroParams2S, Gnet::Array{Float64,3},
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
                    Gnet[i, x[i], m] > 0 && (x[i] = 1)
                end
            end
            # 2-state natural transition
            for i in 1:p.C
                if x[i] == 1 && rand() < p.p_fail[i]
                    x[i] = 2
                end
                # state 2 is absorbing without repair
            end
        end
        total_openings += opens
    end
    return total_openings / nsim
end

function find_lambda_star(p::HeteroParams2S, Gnet::Array{Float64,3};
                           tol=1e-3, maxit=40)
    λmin = 0.0
    λmax = p.H * sum(p.h[i] for i in 1:p.C)
    for _ in 1:maxit
        λ = (λmin + λmax) / 2
        openings = simulate_openings(p, Gnet, λ; nsim=500)
        g = openings - p.K
        abs(g) < tol && break
        g > 0 ? (λmin = λ) : (λmax = λ)
    end
    return (λmin + λmax) / 2
end

function simulate_cycle(p::HeteroParams2S, Gnet::Array{Float64,3},
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
        total_loss += sum(p.h[i] * (x[i] == 2) for i in 1:p.C)

        if Sm > η
            R = [i for i in 1:p.C if Gnet[i, x[i], m] > 0]
            total_cost += sum(p.c_repair[i] for i in R)
            if k >= 1 && λmk <= p.π
                k -= 1
            else
                total_cost += p.π
            end
            for i in R
                rand() < p.p_repair[i] && (x[i] = 1)
            end
        end

        # natural 2-state transitions
        for i in 1:p.C
            x[i] == 1 && rand() < p.p_fail[i] && (x[i] = 2)
        end
    end
    return total_loss + total_cost
end

# ============================================================================
# SHARED UTILITY: per-component transition matrices
# ============================================================================

"""
Trans[i, s′, s, a] = P(next state = s′ | current state = s, action a).

No repair (a=1): 1→{1,2}, 2→{2} (absorbing).
Repair   (a=2): with prob p_repair → state 1; with prob (1−p_repair) → natural.
"""
function build_trans(p::HeteroParams2S)
    Trans = zeros(p.C, 2, 2, 2)
    for i in 1:p.C
        # no repair (a=1)
        Trans[i, 1, 1, 1] = 1 - p.p_fail[i];  Trans[i, 2, 1, 1] = p.p_fail[i]
        Trans[i, 2, 2, 1] = 1.0               # state 2 absorbing

        # repair (a=2): succeed → state 1; fail → natural transition
        for s in 1:2, sp in 1:2
            Trans[i, sp, s, 2] = p.p_repair[i] * (sp == 1 ? 1.0 : 0.0) +
                                  (1 - p.p_repair[i]) * Trans[i, sp, s, 1]
        end
    end
    return Trans
end

# ============================================================================
# HEURISTIC 2: ROLLING HORIZON LP (soft budget in expectation)
# ============================================================================

function fluid_lp_rolling_horizon(p::HeteroParams2S, x_current::Vector{Int}, m::Int)
    m <= 0 && return nothing, nothing, 0.0
    Trans = build_trans(p)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[1:p.C, 1:2, 1:2, 0:1, 1:m] >= 0)
    @variable(model, 0 <= w_t[1:m] <= 1)
    @variable(model, y >= 0)

    # initial condition
    for i in 1:p.C
        s_i = x_current[i]
        @constraint(model, sum(x[i, s_i, a, ww, m] for a in 1:2, ww in 0:1) == 1.0)
        for s in 1:2
            s != s_i && @constraint(model,
                sum(x[i, s, a, ww, m] for a in 1:2, ww in 0:1) == 0.0)
        end
    end

    # repair requires a window
    for t in 1:m, i in 1:p.C, s in 1:2
        @constraint(model, x[i, s, 2, 0, t] == 0.0)
    end

    # Kolmogorov flow
    for t in 1:m-1, i in 1:p.C, sp in 1:2
        flow_to   = sum(x[i, sp, a, ww, t] for a in 1:2, ww in 0:1)
        flow_from = sum(Trans[i, sp, s, a] * x[i, s, a, ww, t+1]
                        for s in 1:2, a in 1:2, ww in 0:1)
        @constraint(model, flow_to == flow_from)
    end

    # window sharing: equality — shared window probability is the same for all components
    for t in 1:m, i in 1:p.C
        @constraint(model, w_t[t] == sum(x[i, s, a, 1, t] for s in 1:2, a in 1:2))
    end

    # normalisation
    for t in 1:m, i in 1:p.C
        @constraint(model,
            sum(x[i, s, a, ww, t] for s in 1:2, a in 1:2, ww in 0:1) == 1.0)
    end

    # soft budget
    @constraint(model, y >= sum(w_t[t] for t in 1:m) - p.K)

    repair_cost  = sum(p.c_repair[i] * x[i, s, 2, ww, t]
                       for i in 1:p.C, s in 1:2, ww in 0:1, t in 1:m)
    holding_cost = sum(p.h[i] * x[i, 2, a, ww, t]
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

function simulate_cycle_lp_marginal(p::HeteroParams2S; seed=1234)
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
                sum(x_opt[i, s_i, 2, ww, m] for ww in 0:1) > 0.5 && push!(R, i)
            end
            w_m = w_opt[m]
        end

        total_loss += sum(p.h[i] * (x[i] == 2) for i in 1:p.C)

        if !isempty(R)
            total_cost += sum(p.c_repair[i] for i in R)
            if w_m > 0.5 && k >= 1
                k -= 1
            else
                total_cost += p.π
            end
            for i in R
                rand() < p.p_repair[i] && (x[i] = 1)
            end
        end

        for i in 1:p.C
            x[i] == 1 && rand() < p.p_fail[i] && (x[i] = 2)
        end
    end
    return total_loss + total_cost
end

# ============================================================================
# HEURISTIC 3: ROLLING HORIZON LP — k in state, hard window constraint
# ============================================================================

"""
State per component: (s, k) — condition s ∈ {1,2}, remaining free windows k ∈ {0,...,K}.
The free window is shared: if open at period t, k decreases for ALL components
regardless of whether they repair. (a=1, w=1) is therefore valid; only w=1 at k=0
is forbidden.
"""
function fluid_lp_deterministic_windows(p::HeteroParams2S,
                                         x_current::Vector{Int},
                                         k_current::Int, m::Int)
    m <= 0 && return nothing, nothing, nothing, 0.0
    K     = p.K
    Trans = build_trans(p)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[1:p.C, 1:2, 0:K, 1:2, 0:1, 1:m] >= 0)
    @variable(model, 0 <= w_t_free[0:K, 1:m] <= 1)
    @variable(model, 0 <= w_t_paid[0:K, 1:m] <= 1)

    # initial condition
    for i in 1:p.C
        s_i = x_current[i]
        @constraint(model,
            sum(x[i, s_i, k_current, a, w, m] for a in 1:2, w in 0:1) == 1.0)
        for s in 1:2, k in 0:K
            if s != s_i || k != k_current
                @constraint(model,
                    sum(x[i, s, k, a, w, m] for a in 1:2, w in 0:1) == 0.0)
            end
        end
    end

    # w=1 forbidden when k=0
    for t in 1:m, i in 1:p.C, s in 1:2
        @constraint(model, x[i, s, 0, 2, 1, t] == 0.0)
        @constraint(model, x[i, s, 0, 1, 1, t] == 0.0)
    end

    # repair requires a window
    for t in 1:m, i in 1:p.C, s in 1:2, k in 0:K
        @constraint(model, x[i, s, k, 2, 0, t] == 0.0)
    end

    # Kolmogorov flow: w=0 → k unchanged; w=1 → k decreases by 1
    for t in 1:m-1, i in 1:p.C, sp in 1:2, kp in 0:K
        flow_to   = sum(x[i, sp, kp, a, w, t] for a in 1:2, w in 0:1)
        flow_from = sum(Trans[i, sp, s, a] * x[i, s, kp, a, 0, t+1]
                        for s in 1:2, a in 1:2)
        if kp + 1 <= K
            flow_from += sum(Trans[i, sp, s, a] * x[i, s, kp+1, a, 1, t+1]
                             for s in 1:2, a in 1:2)
        end
        @constraint(model, flow_to == flow_from)
    end

    # window sharing: equality — shared window probability is the same for all components,
    # conditioning on each budget level k separately.
    for t in 1:m, i in 1:p.C, k in 0:K
        @constraint(model,
            w_t_free[k, t] == sum(x[i, s, k, a, 1, t] for s in 1:2, a in 1:2))
        @constraint(model,
            w_t_paid[k, t] == sum(x[i, s, k, 2, 0, t] for s in 1:2))
    end

    # normalisation
    for t in 1:m, i in 1:p.C
        @constraint(model,
            sum(x[i, s, k, a, w, t]
                for s in 1:2, k in 0:K, a in 1:2, w in 0:1) == 1.0)
    end

    repair_cost  = sum(p.c_repair[i] * x[i, s, k, 2, w, t]
                       for i in 1:p.C, s in 1:2, k in 0:K, w in 0:1, t in 1:m)
    holding_cost = sum(p.h[i] * x[i, 2, k, a, w, t]
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

function simulate_cycle_lp_deterministic(p::HeteroParams2S; seed=1234)
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
                # repair = a=2 with any window type (w=0 is ruled out by LP constraint)
                sum(x_opt[i, s_i, k, 2, w, m] for w in 0:1) > 0.5 && push!(R, i)
            end
            use_free = k > 0 && w_free_opt[k, m] >= 0.5
        end

        total_loss += sum(p.h[i] * (x[i] == 2) for i in 1:p.C)

        if !isempty(R)
            total_cost += sum(p.c_repair[i] for i in R)
            if use_free
                k -= 1
            else
                total_cost += p.π
            end
            for i in R
                rand() < p.p_repair[i] && (x[i] = 1)
            end
        end

        for i in 1:p.C
            x[i] == 1 && rand() < p.p_fail[i] && (x[i] = 2)
        end
    end
    return total_loss + total_cost
end

# ============================================================================
# LP LOWER BOUNDS (full horizon, all-healthy start, k = K)
# ============================================================================

lp_lower_bound_rolling(p::HeteroParams2S) =
    fluid_lp_rolling_horizon(p, fill(1, p.C), p.H)[3]

lp_lower_bound_deterministic(p::HeteroParams2S) =
    fluid_lp_deterministic_windows(p, fill(1, p.C), p.K, p.H)[4]

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

function run_experiment(; seed_params=42, nsim=500)
    println("\n" * "="^80)
    println("HETEROGENEOUS MAINTENANCE — 2 STATES (good / failed)")
    println("="^80)

    rng = Random.MersenneTwister(seed_params)
    C, H, K = 5, 24, 3

    p = HeteroParams2S(
        C        = C,
        H        = H,
        K        = K,
        p_fail   = rand(rng, C) .* 0.25 .+ 0.05,   # U[0.05, 0.30]
        p_repair = fill(1.0, C),                    # perfect repair for now
        h        = rand(rng, C) .* 2.00 .+ 0.50,   # U[0.50, 2.50]
        c_repair = rand(rng, C) .* 0.30 .+ 0.05,   # U[0.05, 0.35]
        π        = 5.0
    )

    println("\nGlobal: C=$C  H=$H  K=$K  π=$(p.π)")
    println("\n  Comp │ p_fail │ p_repair │   h    │ c_repair")
    println("  ─────┼────────┼──────────┼────────┼─────────")
    for i in 1:C
        @printf("    %d  │  %.3f │  %.3f   │  %.3f │  %.3f\n",
                i, p.p_fail[i], p.p_repair[i], p.h[i], p.c_repair[i])
    end

    # LP lower bounds
    println("\n" * "-"^80)
    println("LP LOWER BOUNDS (full horizon, all-healthy start)")
    println("-"^80)
    lb_rolling = lp_lower_bound_rolling(p)
    println("  LP rolling  (soft budget)    : ", round(lb_rolling, digits=4))
    lb_det = lp_lower_bound_deterministic(p)
    println("  LP det.     (k in state)     : ", round(lb_det,     digits=4))

    # Heuristic 1
    println("\n" * "-"^80)
    println("HEURISTIC 1: NET GAINS + SHADOW PRICE")
    println("-"^80)
    u = surrogate_values(p)
    _, Gnet = repair_gains(p, u)
    λstar = find_lambda_star(p, Gnet)
    println("λ* ≈ ", round(λstar, digits=3))
    costs_h1 = [simulate_cycle(p, Gnet, λstar; seed=s) for s in 1:nsim]
    println("Results ($nsim sims):  mean=$(round(mean(costs_h1),digits=4))  std=$(round(std(costs_h1),digits=4))")


    # Heuristic 3
    println("\n" * "-"^80)
    println("HEURISTIC 3: ROLLING LP (k in state, hard window constraint)")
    println("-"^80)
    costs_h3 = [simulate_cycle_lp_deterministic(p; seed=s) for s in 1:nsim]
    println("Results ($nsim sims):  mean=$(round(mean(costs_h3),digits=4))  std=$(round(std(costs_h3),digits=4))")

    # Summary
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

run_experiment()

end # module HeteroMaintenance2S
