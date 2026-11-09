module FreeWindowsHetero3S

using Random, Statistics, Printf
using JuMP, HiGHS
using Base: @kwdef

# States: 1 = good, 2 = degraded, 3 = panne (absorbing without repair)
#
# Problem: K free maintenance windows available over H periods.
# NO paid windows — once the K free windows are spent, no further repair
# is possible. Opening a free window costs nothing; repairing component i
# inside an open window costs c_repair[i].
#
# Cost per period:
#   Σ_i h_deg[i]·1{x_i=2} + h_loss[i]·1{x_i=3}   (holding, charged first)
#   + Σ_{i∈R} c_repair[i]                           (marginal repair, if window open)

@kwdef struct FreeWinParams3S
    C::Int
    H::Int
    K::Int
    p_deg::Vector{Float64}    # prob 1→2 per period          (length C)
    p_fail::Vector{Float64}   # prob 2→3 per period          (length C)
    p_repair::Vector{Float64} # repair success probability   (length C)
    h_deg::Vector{Float64}    # holding cost state 2         (length C)
    h_loss::Vector{Float64}   # holding cost state 3         (length C)
    c_repair::Vector{Float64} # marginal repair cost         (length C)
end

# ============================================================================
# SURROGATE VALUES — Option A (no-repair baseline)
# ============================================================================
#
# With no paid-repair option, the single-asset surrogate is the cost-to-go
# under no future maintenance. Option B is not applicable (standalone repair
# cost is infinite).

function surrogate_values(p::FreeWinParams3S)
    u = zeros(p.C, 3, p.H + 1)
    for i in 1:p.C
        for m in 2:p.H+1
            u[i, 1, m] = (1 - p.p_deg[i])  * u[i, 1, m-1] + p.p_deg[i]  * u[i, 2, m-1]
            u[i, 2, m] = p.h_deg[i]  + (1 - p.p_fail[i]) * u[i, 2, m-1] + p.p_fail[i]  * u[i, 3, m-1]
            u[i, 3, m] = p.h_loss[i] + u[i, 3, m-1]
        end
    end
    return u
end

# ============================================================================
# REPAIR GAINS
# ============================================================================

function repair_gains(p::FreeWinParams3S, u::Array{Float64,3})
    G    = zeros(p.C, 3, p.H + 1)
    Gnet = zeros(p.C, 3, p.H + 1)
    for i in 1:p.C, m in 2:p.H+1
        E_wait = [
            (1 - p.p_deg[i])  * u[i, 1, m-1] + p.p_deg[i]  * u[i, 2, m-1],
            (1 - p.p_fail[i]) * u[i, 2, m-1] + p.p_fail[i] * u[i, 3, m-1],
            u[i, 3, m-1]
        ]
        E_rep = [p.p_repair[i] * u[i, 1, m-1] + (1 - p.p_repair[i]) * E_wait[s]
                 for s in 1:3]
        for s in 1:3
            hold          = s == 2 ? p.h_deg[i] : (s == 3 ? p.h_loss[i] : 0.0)
            G[i, s, m]    = hold + E_wait[s] - E_rep[s]
            Gnet[i, s, m] = G[i, s, m] - p.c_repair[i]
        end
    end
    return G, Gnet
end

# ============================================================================
# HEURISTIC 1 — Shadow-price (free windows only)
# ============================================================================
#
# λ* is calibrated so that the expected number of free-window openings = K.
# Online rule with m periods remaining and k free windows left:
#   threshold η = λ* · (K/H) / (k/m)   (tighten when budget is tight)
#   Open window iff k > 0  AND  Σ_i [Gnet_i,m(x_i)]⁺ > η
# (No paid fallback: k=0 means no maintenance for the rest of the horizon.)

function lambda_adjust(λstar, m, k, p::FreeWinParams3S)
    k == 0  && return Inf          # budget exhausted → never open
    k >= m  && return 0.0          # more budget than periods left → always open if any gain
    return λstar * (p.K / p.H) / (k / m)
end

function simulate_openings(p::FreeWinParams3S, Gnet::Array{Float64,3},
                            λ::Float64; nsim::Int=500)
    total_openings = 0
    for _ in 1:nsim
        x = fill(1, p.C); opens = 0
        for m in p.H:-1:1
            Sm = sum(max(Gnet[i, x[i], m], 0.0) for i in 1:p.C)
            if Sm > λ
                opens += 1
                for i in 1:p.C; Gnet[i, x[i], m] > 0 && (x[i] = 1); end
            end
            for i in 1:p.C
                if x[i] == 1 && rand() < p.p_deg[i]; x[i] = 2
                elseif x[i] == 2 && rand() < p.p_fail[i]; x[i] = 3; end
            end
        end
        total_openings += opens
    end
    return total_openings / nsim
end

function find_lambda_star(p::FreeWinParams3S, Gnet::Array{Float64,3};
                           tol=1e-3, maxit=40)
    λmin, λmax = 0.0, p.H * sum(p.h_loss[i] + p.h_deg[i] for i in 1:p.C)
    for _ in 1:maxit
        λ = (λmin + λmax) / 2
        g = simulate_openings(p, Gnet, λ; nsim=500) - p.K
        abs(g) < tol && break
        g > 0 ? (λmin = λ) : (λmax = λ)
    end
    return (λmin + λmax) / 2
end

function simulate_h1(p::FreeWinParams3S, Gnet::Array{Float64,3},
                     λstar::Float64; seed=1234)
    Random.seed!(seed)
    k = p.K; total_loss = 0.0; total_cost = 0.0
    x = fill(1, p.C)
    for m in p.H:-1:1
        Sm  = sum(max(Gnet[i, x[i], m], 0.0) for i in 1:p.C)
        η   = lambda_adjust(λstar, m, k, p)

        total_loss += sum(p.h_deg[i]*(x[i]==2) + p.h_loss[i]*(x[i]==3) for i in 1:p.C)

        if k > 0 && Sm > η
            R = [i for i in 1:p.C if Gnet[i, x[i], m] > 0]
            total_cost += sum(p.c_repair[i] for i in R)
            k -= 1
            for i in R; rand() < p.p_repair[i] && (x[i] = 1); end
        end
        for i in 1:p.C
            if x[i] == 1 && rand() < p.p_deg[i]; x[i] = 2
            elseif x[i] == 2 && rand() < p.p_fail[i]; x[i] = 3; end
        end
    end
    return total_loss + total_cost
end

# ============================================================================
# TRANSITION MATRICES
# ============================================================================

function build_trans(p::FreeWinParams3S)
    Trans = zeros(p.C, 3, 3, 2)
    for i in 1:p.C
        Trans[i, 1, 1, 1] = 1 - p.p_deg[i];  Trans[i, 2, 1, 1] = p.p_deg[i]
        Trans[i, 2, 2, 1] = 1 - p.p_fail[i]; Trans[i, 3, 2, 1] = p.p_fail[i]
        Trans[i, 3, 3, 1] = 1.0
        for s in 1:3, sp in 1:3
            Trans[i, sp, s, 2] = p.p_repair[i] * (sp == 1 ? 1.0 : 0.0) +
                                  (1 - p.p_repair[i]) * Trans[i, sp, s, 1]
        end
    end
    return Trans
end

# ============================================================================
# HEURISTIC 2 — Rolling LP (hard budget, no paid windows)
# ============================================================================
#
# At each period the LP is solved over the remaining m steps with k_remaining
# free windows. Budget is enforced as a hard constraint Σw_t ≤ k_remaining.
# No paid window variable, no π in the objective.

function rolling_lp_h2(p::FreeWinParams3S, x_current::Vector{Int},
                        k_remaining::Int, m::Int)
    m <= 0 && return nothing, nothing, 0.0
    Trans = build_trans(p)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # x[i, s, k, a, ww, t]: k = remaining free windows (in state);
    # ww=1 means a free window is open this period and k decrements for ALL components,
    # even those that do not repair (a=1). ww is the shared window indicator.
    K = p.K
    @variable(model, x[1:p.C, 1:3, 0:K, 1:2, 0:1, 1:m] >= 0)
    # w_t[k, t] = P(window open at period t AND k free windows remaining).
    # Finer than a single w_t[t]: couples components at each (t, k) pair separately.
    @variable(model, 0 <= w_t[0:K, 1:m] <= 1)

    # ── initial condition: k starts at k_remaining ────────────────────────────
    for i in 1:p.C
        s_i = x_current[i]
        @constraint(model,
            sum(x[i, s_i, k_remaining, a, ww, m] for a in 1:2, ww in 0:1) == 1.0)
        for s in 1:3, k in 0:K
            (s != s_i || k != k_remaining) && @constraint(model,
                sum(x[i, s, k, a, ww, m] for a in 1:2, ww in 0:1) == 0.0)
        end
    end

    # ── ww=1 forbidden when k=0 (no free windows left) ───────────────────────
    for t in 1:m, i in 1:p.C, s in 1:3, a in 1:2
        @constraint(model, x[i, s, 0, a, 1, t] == 0.0)
    end

    # ── repair requires an open window ────────────────────────────────────────
    for t in 1:m, i in 1:p.C, s in 1:3, k in 0:K
        @constraint(model, x[i, s, k, 2, 0, t] == 0.0)
    end

    # ── Kolmogorov flow: ww=0 → k unchanged; ww=1 → k decreases by 1 ─────────
    for t in 1:m-1, i in 1:p.C, sp in 1:3, kp in 0:K
        flow_to   = sum(x[i, sp, kp, a, ww, t] for a in 1:2, ww in 0:1)
        flow_from = sum(Trans[i, sp, s, a] * x[i, s, kp, a, 0, t+1]
                        for s in 1:3, a in 1:2)
        if kp + 1 <= K
            flow_from += sum(Trans[i, sp, s, a] * x[i, s, kp+1, a, 1, t+1]
                             for s in 1:3, a in 1:2)
        end
        @constraint(model, flow_to == flow_from)
    end

    # ── window sharing: w_t[k,t] = P(window open at t, k windows left), per i ─
    # Equality at each (k, t): the shared window probability is the same for all i.
    for t in 1:m, i in 1:p.C, k in 0:K
        @constraint(model,
            w_t[k, t] == sum(x[i, s, k, a, 1, t] for s in 1:3, a in 1:2))
    end

    # ── normalisation ─────────────────────────────────────────────────────────
    for t in 1:m, i in 1:p.C
        @constraint(model,
            sum(x[i, s, k, a, ww, t]
                for s in 1:3, k in 0:K, a in 1:2, ww in 0:1) == 1.0)
    end

    holding = sum(p.h_deg[i]*x[i,2,k,a,ww,t] + p.h_loss[i]*x[i,3,k,a,ww,t]
                  for i in 1:p.C, k in 0:K, a in 1:2, ww in 0:1, t in 1:m)
    repair  = sum(p.c_repair[i]*x[i,s,k,2,ww,t]
                  for i in 1:p.C, s in 1:3, k in 0:K, ww in 0:1, t in 1:m)
    @objective(model, Min, holding + repair)

    optimize!(model)
    status = termination_status(model)
    status ∉ (MOI.OPTIMAL, MOI.FEASIBLE_POINT) && return nothing, nothing, NaN
    return value.(x), value.(w_t), objective_value(model)
end

function simulate_h2(p::FreeWinParams3S; seed=1234)
    Random.seed!(seed)
    k = p.K; total_loss = 0.0; total_cost = 0.0
    x = fill(1, p.C)
    for m in p.H:-1:1
        x_opt, w_opt, _ = rolling_lp_h2(p, x, k, m)
        R = Int[]; open_window = false
        if x_opt !== nothing
            for i in 1:p.C
                s_i = x[i]
                sum(x_opt[i, s_i, k, 2, ww, m] for ww in 0:1) > 0.5 && push!(R, i)
            end
            open_window = k > 0 && w_opt[k, m] >= 0.5
        end
        total_loss += sum(p.h_deg[i]*(x[i]==2) + p.h_loss[i]*(x[i]==3) for i in 1:p.C)
        if !isempty(R) && open_window && k > 0
            total_cost += sum(p.c_repair[i] for i in R)
            k -= 1
            for i in R; rand() < p.p_repair[i] && (x[i] = 1); end
        end
        for i in 1:p.C
            if x[i] == 1 && rand() < p.p_deg[i]; x[i] = 2
            elseif x[i] == 2 && rand() < p.p_fail[i]; x[i] = 3; end
        end
    end
    return total_loss + total_cost
end

# ============================================================================
# HEURISTIC 3 — Rolling LP (k in state, no paid windows)
# ============================================================================
#
# The remaining free-window count k ∈ {0,...,K} is part of each component's
# state (s, k). w=1 is forbidden when k=0. No paid-window variable.

function rolling_lp_h3(p::FreeWinParams3S, x_current::Vector{Int},
                        k_current::Int, m::Int)
    m <= 0 && return nothing, nothing, 0.0
    K = p.K; Trans = build_trans(p)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # x[i, s, k, a, ww, t]: k = remaining free windows (in state);
    # ww=1 means a free window is open this period and k decrements for ALL components,
    # even those that do not repair (a=1). ww is the shared window indicator.
    @variable(model, x[1:p.C, 1:3, 0:K, 1:2, 0:1, 1:m] >= 0)
    # w_t[k, t] = P(window open at period t AND k free windows remaining).
    # Finer than a single w_t[t]: couples components at each (t, k) pair separately.
    @variable(model, 0 <= w_t[0:K, 1:m] <= 1)

    # ── initial condition ─────────────────────────────────────────────────────
    for i in 1:p.C
        s_i = x_current[i]
        @constraint(model,
            sum(x[i, s_i, k_current, a, ww, m] for a in 1:2, ww in 0:1) == 1.0)
        for s in 1:3, k in 0:K
            (s != s_i || k != k_current) && @constraint(model,
                sum(x[i, s, k, a, ww, m] for a in 1:2, ww in 0:1) == 0.0)
        end
    end

    # ── ww=1 forbidden when k=0 (no free windows left) ───────────────────────
    for t in 1:m, i in 1:p.C, s in 1:3, a in 1:2
        @constraint(model, x[i, s, 0, a, 1, t] == 0.0)
    end

    # ── repair requires an open window ────────────────────────────────────────
    for t in 1:m, i in 1:p.C, s in 1:3, k in 0:K
        @constraint(model, x[i, s, k, 2, 0, t] == 0.0)
    end

    # ── Kolmogorov flow ───────────────────────────────────────────────────────
    # ww=0 → k unchanged; ww=1 → k decreases by 1 (window consumed by shared event)
    for t in 1:m-1, i in 1:p.C, sp in 1:3, kp in 0:K
        flow_to   = sum(x[i, sp, kp, a, ww, t] for a in 1:2, ww in 0:1)
        flow_from = sum(Trans[i, sp, s, a] * x[i, s, kp, a, 0, t+1]
                        for s in 1:3, a in 1:2)
        if kp + 1 <= K
            flow_from += sum(Trans[i, sp, s, a] * x[i, s, kp+1, a, 1, t+1]
                             for s in 1:3, a in 1:2)
        end
        @constraint(model, flow_to == flow_from)
    end

    # ── window sharing: w_t[k,t] = P(window open at t, k windows left), per i ─
    # Equality at each (k, t): the shared window probability is the same for all i.
    for t in 1:m, i in 1:p.C, k in 0:K
        @constraint(model,
            w_t[k, t] == sum(x[i, s, k, a, 1, t] for s in 1:3, a in 1:2))
    end

    # ── normalisation ─────────────────────────────────────────────────────────
    for t in 1:m, i in 1:p.C
        @constraint(model,
            sum(x[i, s, k, a, ww, t]
                for s in 1:3, k in 0:K, a in 1:2, ww in 0:1) == 1.0)
    end

    holding = sum(p.h_deg[i]*x[i,2,k,a,ww,t] + p.h_loss[i]*x[i,3,k,a,ww,t]
                  for i in 1:p.C, k in 0:K, a in 1:2, ww in 0:1, t in 1:m)
    repair  = sum(p.c_repair[i]*x[i,s,k,2,ww,t]
                  for i in 1:p.C, s in 1:3, k in 0:K, ww in 0:1, t in 1:m)
    @objective(model, Min, holding + repair)

    optimize!(model)
    status = termination_status(model)
    status ∉ (MOI.OPTIMAL, MOI.FEASIBLE_POINT) && return nothing, nothing, NaN
    return value.(x), value.(w_t), objective_value(model)
end

function simulate_h3(p::FreeWinParams3S; seed=1234)
    println(seed)
    Random.seed!(seed)
    k = p.K; total_loss = 0.0; total_cost = 0.0
    x = fill(1, p.C)
    for m in p.H:-1:1
        x_opt, w_opt, _ = rolling_lp_h3(p, x, k, m)
        R = Int[]; use_free = false
        if x_opt !== nothing
            for i in 1:p.C
                s_i = x[i]
                sum(x_opt[i, s_i, k, 2, ww, m] for ww in 0:1) > 0.5 && push!(R, i)
            end
            use_free = k > 0 && w_opt[k, m] >= 0.5
        end
        total_loss += sum(p.h_deg[i]*(x[i]==2) + p.h_loss[i]*(x[i]==3) for i in 1:p.C)
        if !isempty(R) && use_free
            total_cost += sum(p.c_repair[i] for i in R)
            k -= 1
            for i in R; rand() < p.p_repair[i] && (x[i] = 1); end
        end
        for i in 1:p.C
            if x[i] == 1 && rand() < p.p_deg[i]; x[i] = 2
            elseif x[i] == 2 && rand() < p.p_fail[i]; x[i] = 3; end
        end
    end
    return total_loss + total_cost
end

# ============================================================================
# LP LOWER BOUND
# ============================================================================

function lp_lower_bound(p::FreeWinParams3S)
    _, _, lb = rolling_lp_h3(p, fill(1, p.C), p.K, p.H)
    return lb
end

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

function run_experiment(; seed_params=44, nsim=50)
    println("\n" * "="^80)
    println("FREE WINDOWS ONLY — 3 STATES (good / degraded / panne)")
    println("K free windows, no paid option. Repair inside window costs c_repair[i].")
    println("="^80)

    rng = Random.MersenneTwister(seed_params)
    C, H, K = 5, 24, 3

    p = FreeWinParams3S(
        C        = C, H = H, K = K,
        p_deg    = rand(rng, C) .* 0.20 .+ 0.05,
        p_fail   = rand(rng, C) .* 0.30 .+ 0.05,
        p_repair = fill(1.0, C),
        h_deg    = rand(rng, C) .* 0.50 .+ 0.10,
        h_loss   = rand(rng, C) .* 2.00 .+ 0.50,
        c_repair = rand(rng, C) .* 0.30 .+ 0.05,
    )

    println("\nGlobal: C=$C  H=$H  K=$K  (no paid windows)")
    println("\n  Comp │ p_deg │ p_fail │ p_repair │ h_deg │ h_loss │ c_repair")
    println("  ─────┼───────┼────────┼──────────┼───────┼────────┼─────────")
    for i in 1:C
        @printf("    %d  │ %.3f │  %.3f  │  %.3f   │ %.3f │  %.3f  │  %.3f\n",
                i, p.p_deg[i], p.p_fail[i], p.p_repair[i],
                p.h_deg[i], p.h_loss[i], p.c_repair[i])
    end

    println("\n" * "-"^80)
    println("LP LOWER BOUND"); println("-"^80)
    lb = lp_lower_bound(p)
    println("  LP (hard budget) : ", round(lb, digits=4))

    println("\n" * "-"^80)
    println("H1: NET GAINS + SHADOW PRICE"); println("-"^80)
    u = surrogate_values(p)
    _, Gnet = repair_gains(p, u)
    λstar = find_lambda_star(p, Gnet)
    @printf("  λ* ≈ %.4f\n", λstar)
    costs_h1 = [simulate_h1(p, Gnet, λstar; seed=s) for s in 1:nsim]
    @printf("  Results (%d sims): mean=%.4f  std=%.4f\n", nsim, mean(costs_h1), std(costs_h1))

    #println("\n" * "-"^80)
    #println("H2: ROLLING LP (hard budget, re-solved every period)"); println("-"^80)
    #costs_h2 = [simulate_h2(p; seed=s) for s in 1:nsim]
    #@printf("  Results (%d sims): mean=%.4f  std=%.4f\n", nsim, mean(costs_h2), std(costs_h2))

    println("\n" * "-"^80)
    println("H3: ROLLING LP (k in state)"); println("-"^80)
    costs_h3 = [simulate_h3(p; seed=s) for s in 1:nsim]
    @printf("  Results (%d sims): mean=%.4f  std=%.4f\n", nsim, mean(costs_h3), std(costs_h3))

    println("\n" * "="^80); println("SUMMARY"); println("="^80)
    @printf("  %-30s  %8s  %10s\n", "Method", "Mean", "Gap vs LB")
    println("  " * "─"^52)
    @printf("  %-30s  %8.4f\n", "LP lower bound", lb)
    for (label, costs) in [("H1 (shadow price)",  costs_h1),
                            ("H3 (k in state)",    costs_h3)]
        @printf("  %-30s  %8.4f  %+6.1f%%\n", label, mean(costs), (mean(costs)/lb-1)*100)
    end
    println("="^80)
end

run_experiment()

end # module FreeWindowsHetero3S
