using Random, Statistics
using JuMP, HiGHS

using Base: @kwdef

@kwdef struct MaintenanceParams
    C::Int
    H::Int
    K::Int
    p_fail::Float64
    p_repair::Float64
    h_loss::Float64
    c_repair::Float64
    π::Float64
end

# ============================================================================
# SURROGATE VALUES & GAINS
# ============================================================================

function surrogate_values(p::MaintenanceParams)
    u = zeros(2, p.H + 1)
    for m in 2:p.H+1
        u[1, m] = p.p_fail * (p.h_loss + u[2, m-1]) + (1 - p.p_fail) * u[1, m-1]
        u[2, m] = p.h_loss + u[2, m-1]
    end
    return u
end

function repair_gains(p::MaintenanceParams, u::Array{Float64,2})
    G = zeros(2, p.H + 1)
    for m in 2:p.H+1
        G[2, m] = p.h_loss + u[2, m-1] - u[1, m-1]
        G[1, m] = 0.0
    end
    Gnet = G .- [0.0; p.c_repair]
    return G, Gnet
end

function lambda_adjust(λstar, m, k, p)
    k >= m ? 0.0 : (k == 0 ? Inf : λstar * (p.K / p.H) / (k / m))
end

# ============================================================================
# HEURISTIC 1: GAINS NET + SHADOW PRICE
# ============================================================================

function simulate_openings(p::MaintenanceParams, Gnet::Array{Float64,2}, λ::Float64; nsim::Int=500)
    total_openings = 0
    for _ in 1:nsim
        x = fill(1, p.C)
        opens = 0
        for m in p.H:-1:1
            Sm = sum(Gnet[x[i], m] > 0 ? Gnet[x[i], m] : 0.0 for i in 1:p.C)
            if Sm > λ
                opens += 1
                for i in 1:p.C
                    if Gnet[x[i], m] > 0
                        x[i] = 1
                    end
                end
            end
            for i in 1:p.C
                if x[i] == 1 && rand() < p.p_fail
                    x[i] = 2
                end
            end
        end
        total_openings += opens
    end
    return total_openings / nsim
end

function find_lambda_star(p::MaintenanceParams, Gnet::Array{Float64,2}; tol=1e-3, maxit=40)
    λmin, λmax = 0.0, p.H * p.C * p.h_loss
    for _ in 1:maxit
        λ = (λmin + λmax) / 2
        openings = simulate_openings(p, Gnet, λ; nsim=500)
        g = openings - p.K
        abs(g) < tol && break
        g > 0 ? (λmin = λ) : (λmax = λ)
    end
    return (λmin + λmax) / 2
end

function simulate_cycle(p::MaintenanceParams, Gnet::Array{Float64,2}, λstar::Float64; seed=1234)
    Random.seed!(seed)
    k = p.K
    total_loss = 0.0
    total_cost = 0.0
    x = fill(1, p.C)
    
    for m in p.H:-1:1
        Sm = sum(Gnet[x[i], m] > 0 ? Gnet[x[i], m] : 0.0 for i in 1:p.C)
        λmk = lambda_adjust(λstar, m, k, p)
        η = k >= 1 ? min(λmk, p.π) : p.π
        
        R = Int[]
        if Sm > η
            R = [i for i in 1:p.C if Gnet[x[i], m] > 0]
            total_cost += length(R) * p.c_repair
            if k >= 1 && λmk <= p.π
                k -= 1
            else
                total_cost += p.π
            end
            for i in R
                x[i] = 1
            end
        end
        
        total_loss += p.h_loss * count(x[i] == 2 for i in 1:p.C)
        for i in 1:p.C
            if x[i] == 1 && rand() < p.p_fail
                x[i] = 2
            end
        end
    end
    
    return total_loss + total_cost
end

# ============================================================================
# ROLLING HORIZON LP WITH WINDOW VARIABLES
# ============================================================================

"""
Solve fluid LP with rolling horizon m periods remaining.

VARIABLES:
  x[i, s, a, w, t] = prob that component i is in state s, takes action a,
                     uses window w (0=no, 1=yes), at period t
  w_t[t] = probability that window is opened at period t (COMMON to all components)

KEY INSIGHT:
  - Each component has its own window variable x[i, :, :, w, t]
  - But all components share the SAME w_t[t]
  - Constraint: x[i, s, 2, w, t] = 0 when w=0 (repair needs window)
  - Constraint: w_t[t] >= max_i(prob component i uses window)
  
This incentivizes:
  - If opening window for one, repair as many as possible!
  - Window cost is shared across multiple repairs
"""
function fluid_lp_rolling_horizon(p::MaintenanceParams, x_current::Vector{Int}, m::Int)
    if m <= 0
        return nothing, nothing, 0.0
    end
    
    Trans = zeros(2, 2, 2)
    Trans[1, 1, 1] = 1 - p.p_fail;  Trans[2, 1, 1] = p.p_fail
    Trans[1, 2, 1] = 0.0;           Trans[2, 2, 1] = 1.0
    for s in 1:2
        Trans[1, s, 2] = p.p_repair * (1 - p.p_fail)
        Trans[2, s, 2] = 1 - p.p_repair * (1 - p.p_fail)
    end
 
    model = Model(HiGHS.Optimizer)
    set_silent(model)
 
    # Variables with window dimension
    @variable(model, x[1:p.C, 1:2, 1:2, 0:1, 1:m] >= 0)
    @variable(model, 0 <= w_t[1:m] <= 1)
    @variable(model, y >= 0)
 
    # ========== BOUNDARY: CURRENT STATE ==========
    for i in 1:p.C
        s_current = x_current[i]
        @constraint(model, sum(x[i, s_current, a, ww, m] for a in 1:2, ww in 0:1) == 1.0)
        for s in 1:2
            if s != s_current
                @constraint(model, sum(x[i, s, a, ww, m] for a in 1:2, ww in 0:1) == 0.0)
            end
        end
    end
 
    # ========== REPAIR REQUIRES WINDOW ==========
    # If component repairs (a=2), it MUST use window (w=1)
    for t in 1:m, i in 1:p.C, s in 1:2
        @constraint(model, x[i, s, 2, 0, t] == 0.0)
    end
 
    # ========== KOLMOGOROV FLOW ==========
    for t in 1:m-1, i in 1:p.C, sp in 1:2
        flow_to = sum(x[i, sp, a, ww, t] for a in 1:2, ww in 0:1)
        flow_from = sum(Trans[sp, s, a] * x[i, s, a, ww, t+1] for s in 1:2, a in 1:2, ww in 0:1)
        @constraint(model, flow_to == flow_from)
    end
 
    # ========== WINDOW CONSISTENCY: W_T >= MAX USE ==========
    # Window must open if ANY component uses it
    for t in 1:m, i in 1:p.C
        uses_window = sum(x[i, s, 2, 1, t] for s in 1:2)
        @constraint(model, w_t[t] >= uses_window)
    end
 
    # ========== NORMALISATION ==========
    for t in 1:m, i in 1:p.C
        @constraint(model, sum(x[i, s, a, ww, t] for s in 1:2, a in 1:2, ww in 0:1) == 1.0)
    end
 
    # ========== BUDGET ==========
    @constraint(model, y >= sum(w_t[t] for t in 1:m) - p.K)
 
    # ========== OBJECTIVE ==========
    repair_cost = p.c_repair * sum(x[i, s, 2, ww, t] for i in 1:p.C, s in 1:2, ww in 0:1, t in 1:m)
    holding_cost = p.h_loss * sum(x[i, 2, a, ww, t] for i in 1:p.C, a in 1:2, ww in 0:1, t in 1:m)
    window_cost = p.π * y
    
    @objective(model, Min, repair_cost + holding_cost + window_cost)
 
    optimize!(model)
 
    status = termination_status(model)
    if status != MOI.OPTIMAL && status != MOI.FEASIBLE_POINT
        @warn "LP(m=$m) failed: $status"
        return nothing, nothing, NaN
    end
 
    return value.(x), value.(w_t), objective_value(model)
end

# ============================================================================
# HEURISTIC 2: ROLLING HORIZON (FINAL VERSION)
# ============================================================================

function simulate_cycle_lp_marginal(p::MaintenanceParams; seed=1234)
    Random.seed!(seed)
    k = p.K
    total_loss = 0.0
    total_cost = 0.0
    x = fill(1, p.C)
    
    for m in p.H:-1:1
        x_opt, w_opt, lb = fluid_lp_rolling_horizon(p, x, m)
        
        R = Int[]
        w_m = 0.0
        
        if x_opt !== nothing && w_opt !== nothing
            # Extract repair prob for each component (marginalize over window)
            repair_prob = zeros(p.C)
            for i in 1:p.C
                s_current = x[i]
                # repair_prob[i] = Σ_w x[i, s_current, 2, w, m]
                repair_prob[i] = sum(x_opt[i, s_current, 2, ww, m] for ww in 0:1)
            end
            
            # Collect components with repair_prob > 0.5
            for i in 1:p.C
                if repair_prob[i] > 0.5
                    push!(R, i)
                end
            end
            
            w_m = w_opt[m]
        end
        
        if !isempty(R)
            total_cost += length(R) * p.c_repair
            if w_m > 0.5 && k >= 1
                k -= 1
            else
                total_cost += p.π
            end
            for i in R
                x[i] = 1
            end
        end
        
        total_loss += p.h_loss * count(x[i] == 2 for i in 1:p.C)
        for i in 1:p.C
            if x[i] == 1 && rand() < p.p_fail
                x[i] = 2
            end
        end
    end
    
    return total_loss + total_cost
end

# ============================================================================
# HEURISTIC 3: ROLLING HORIZON LP WITH DETERMINISTIC WINDOW TRACKING
# ============================================================================

"""
Modified fluid LP where the number of remaining free windows k is part of
the per-component state, replacing the soft budget constraint with a hard
constraint: x[i, s, 0, 2, 1, t] = 0 (cannot open a free window when k = 0).

State:  (s, k)  — condition s ∈ {1,2}, remaining free windows k ∈ {0,...,K}
Action: (a, w)  — repair a ∈ {1=no, 2=yes}, window type w ∈ {0=paid, 1=free}

Valid (a,w) pairs:
  (1,0): no repair
  (2,0): repair with paid window   (k unchanged, costs π via shared w_t_paid)
  (2,1): repair with free window   (k decreases by 1; forbidden when k = 0)

Paid window cost is shared across components via the auxiliary w_t_paid[t].
"""
function fluid_lp_deterministic_windows(p::MaintenanceParams, x_current::Vector{Int}, k_current::Int, m::Int)
    if m <= 0
        return nothing, nothing, 0.0
    end

    K = p.K

    Trans = zeros(2, 2, 2)
    Trans[1, 1, 1] = 1 - p.p_fail;  Trans[2, 1, 1] = p.p_fail
    Trans[1, 2, 1] = 0.0;           Trans[2, 2, 1] = 1.0
    for s in 1:2
        Trans[1, s, 2] = p.p_repair * (1 - p.p_fail)
        Trans[2, s, 2] = 1 - p.p_repair * (1 - p.p_fail)
    end

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # x[i, s, k, a, w, t]
    @variable(model, x[1:p.C, 1:2, 0:K, 1:2, 0:1, 1:m] >= 0)
    # shared window indicators — one value per period, independent of component
    @variable(model, 0 <= w_t_free[1:m] <= 1)   # free window opened
    @variable(model, 0 <= w_t_paid[1:m] <= 1)   # paid window opened

    # ── INITIAL CONDITION at t = m ────────────────────────────────────────────
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

    # ── INVALID ACTIONS ───────────────────────────────────────────────────────
    # The free window is a shared event: if it is open, k decreases for ALL
    # components regardless of whether they repair. So (a=1, w=1) is valid.
    # Only forbidden: free window open when k = 0.
    for t in 1:m, i in 1:p.C, s in 1:2
        @constraint(model, x[i, s, 0, 2, 1, t] == 0.0)   # k=0 ⟹ no free window
        @constraint(model, x[i, s, 0, 1, 1, t] == 0.0)   # k=0 ⟹ no free window (even if no repair)
    end

    # ── KOLMOGOROV FLOW ───────────────────────────────────────────────────────
    # t indexes steps remaining: t = m is the start, t = 1 is the last step.
    # Transition from t+1 → t:
    #   w=0 (paid or no action): k unchanged
    #   w=1 (free window):       k decreases by 1  (kp+1 → kp)
    for t in 1:m-1, i in 1:p.C, sp in 1:2, kp in 0:K
        flow_to   = sum(x[i, sp, kp, a, w, t] for a in 1:2, w in 0:1)
        flow_from = sum(Trans[sp, s, a] * x[i, s, kp, a, 0, t+1]
                        for s in 1:2, a in 1:2)
        if kp + 1 <= K
            flow_from += sum(Trans[sp, s, a] * x[i, s, kp+1, a, 1, t+1]
                             for s in 1:2, a in 1:2)
        end
        @constraint(model, flow_to == flow_from)
    end

    # ── WINDOW SHARING ────────────────────────────────────────────────────────
    # The free/paid window opening decision is shared across all components:
    # w_t_free[t] and w_t_paid[t] must be ≥ any individual component's usage
    # probability, so the LP cannot exploit per-component independence.
    for t in 1:m, i in 1:p.C
        # free window is open whenever w=1, regardless of whether i repairs
        @constraint(model,
            w_t_free[t] >= sum(x[i, s, k, a, 1, t] for s in 1:2, k in 1:K, a in 1:2))
        # paid window is open only when i repairs with w=0
        @constraint(model,
            w_t_paid[t] >= sum(x[i, s, k, 2, 0, t] for s in 1:2, k in 0:K))
    end

    # ── NORMALIZATION ─────────────────────────────────────────────────────────
    for t in 1:m, i in 1:p.C
        @constraint(model,
            sum(x[i, s, k, a, w, t]
                for s in 1:2, k in 0:K, a in 1:2, w in 0:1) == 1.0)
    end

    # ── OBJECTIVE ─────────────────────────────────────────────────────────────
    repair_cost  = p.c_repair * sum(x[i, s, k, 2, w, t]
                       for i in 1:p.C, s in 1:2, k in 0:K, w in 0:1, t in 1:m)
    holding_cost = p.h_loss  * sum(x[i, 2, k, a, w, t]
                       for i in 1:p.C, k in 0:K, a in 1:2, w in 0:1, t in 1:m)
    window_cost  = p.π       * sum(w_t_paid[t] for t in 1:m)

    @objective(model, Min, repair_cost + holding_cost + window_cost)

    optimize!(model)

    status = termination_status(model)
    if status != MOI.OPTIMAL && status != MOI.FEASIBLE_POINT
        @warn "LP_det(m=$m, k=$k_current) failed: $status"
        return nothing, nothing, NaN
    end

    return value.(x), value.(w_t_free), value.(w_t_paid), objective_value(model)
end

# ============================================================================
# HEURISTIC 3: SIMULATION
# ============================================================================

function simulate_cycle_lp_deterministic(p::MaintenanceParams; seed=1234)
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
            repair_p = zeros(p.C)
            for i in 1:p.C
                s_i          = x[i]
                repair_p[i]  = x_opt[i, s_i, k, 2, 1, m] + x_opt[i, s_i, k, 2, 0, m]
            end

            for i in 1:p.C
                if repair_p[i] > 0.5
                    push!(R, i)
                end
            end

            # Window type: use the shared w_t_free[m] signal (component-independent)
            use_free = k > 0 && w_free_opt[m] >= 0.5
        end

        if !isempty(R)
            total_cost += length(R) * p.c_repair
            if use_free
                k -= 1
            else
                total_cost += p.π
            end
            for i in R
                x[i] = 1
            end
        end

        total_loss += p.h_loss * count(x[i] == 2 for i in 1:p.C)
        for i in 1:p.C
            if x[i] == 1 && rand() < p.p_fail
                x[i] = 2
            end
        end
    end

    return total_loss + total_cost
end

# ============================================================================
# LP LOWER BOUNDS (full horizon H, all-healthy start, k = K)
# ============================================================================

"""Return the LP relaxation value from fluid_lp_rolling_horizon at m=H."""
function lp_lower_bound_rolling(p::MaintenanceParams)
    _, _, lb = fluid_lp_rolling_horizon(p, fill(1, p.C), p.H)
    return lb
end

"""Return the LP relaxation value from fluid_lp_deterministic_windows at m=H."""
function lp_lower_bound_deterministic(p::MaintenanceParams)
    _, _, _, lb = fluid_lp_deterministic_windows(p, fill(1, p.C), p.K, p.H)
    return lb
end

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

function run_experiment()
    println("\n" * "="^80)
    println("MAINTENANCE SCHEDULING: THREE HEURISTICS + LP BOUNDS")
    println("="^80)

    p = MaintenanceParams(C=5, H=24, K=3, p_fail=0.2,
                          p_repair=1.0, h_loss=1.0,
                          c_repair=0.1, π=5.0)

    println("\nParameters: C=", p.C, " H=", p.H, " K=", p.K)
    println("  p_fail=", p.p_fail, " h_loss=", p.h_loss)
    println("  c_repair=", p.c_repair, " π=", p.π)

    # LP LOWER BOUNDS
    println("\n" * "-"^80)
    println("LP LOWER BOUNDS (full horizon, all-healthy start)")
    println("-"^80)
    lb_rolling = lp_lower_bound_rolling(p)
    println("  LP rolling (soft budget)      : ", round(lb_rolling, digits=4))
    lb_det = lp_lower_bound_deterministic(p)
    println("  LP det. windows (k in state)  : ", round(lb_det, digits=4))

    # HEURISTIC 1
    println("\n" * "-"^80)
    println("HEURISTIC 1: NET GAINS + SHADOW PRICE")
    println("-"^80)
    u = surrogate_values(p)
    _, Gnet = repair_gains(p, u)
    λstar = find_lambda_star(p, Gnet)
    println("λ* ≈ ", round(λstar, digits=3))
    costs_h1 = [simulate_cycle(p, Gnet, λstar; seed=s) for s in 1:500]
    println("Results (500 sims):")
    println("  Mean : ", round(mean(costs_h1), digits=4))
    println("  Std  : ", round(std(costs_h1), digits=4))

    # HEURISTIC 2
    println("\n" * "-"^80)
    println("HEURISTIC 2: ROLLING LP (soft budget, k in expectation)")
    println("-"^80)
    costs_h2 = [simulate_cycle_lp_marginal(p; seed=s) for s in 1:500]
    println("Results (500 sims):")
    println("  Mean : ", round(mean(costs_h2), digits=4))
    println("  Std  : ", round(std(costs_h2), digits=4))

    # HEURISTIC 3
    println("\n" * "-"^80)
    println("HEURISTIC 3: ROLLING LP (k in state, hard window constraint)")
    println("-"^80)
    costs_h3 = [simulate_cycle_lp_deterministic(p; seed=s) for s in 1:500]
    println("Results (500 sims):")
    println("  Mean : ", round(mean(costs_h3), digits=4))
    println("  Std  : ", round(std(costs_h3), digits=4))

    # SUMMARY
    println("\n" * "="^80)
    println("SUMMARY")
    println("="^80)
    println("  LP bound (rolling)            : ", round(lb_rolling, digits=4))
    println("  LP bound (det. windows)       : ", round(lb_det, digits=4))
    for (label, costs) in [("H1 (gains+shadow)", costs_h1),
                            ("H2 (rolling LP)  ", costs_h2),
                            ("H3 (det. windows)", costs_h3)]
        gap = (mean(costs) / lb_rolling - 1) * 100
        println("  $label mean=$(round(mean(costs),digits=4))  gap=$(round(gap,digits=1))% vs LP-rolling")
    end
    println("="^80)
end

run_experiment()
