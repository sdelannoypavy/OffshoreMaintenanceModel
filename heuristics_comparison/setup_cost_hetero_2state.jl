module SetupCostHetero2S

using Random, Statistics, Printf
using JuMP, HiGHS
using Base: @kwdef

# States: 1 = good, 2 = failed (absorbing without repair)
#
# Problem: no free / paid window budget. Each period, the planner chooses
# which components to repair. Whenever at least one component is repaired,
# a fixed setup cost π is charged (once per period, regardless of how many
# components are maintained). Each repaired component also incurs its own
# marginal cost c_repair[i].
#
# Cost per period:
#   Σ_i h[i]·1{x_i=2}      (holding, charged first)
#   + π · 1{R ≠ ∅}          (setup)
#   + Σ_{i∈R} c_repair[i]   (marginal repair)

@kwdef struct SetupCostParams2S
    C::Int
    H::Int
    p_fail::Vector{Float64}   # prob 1 → 2 per period          (length C)
    p_repair::Vector{Float64} # repair success probability      (length C)
    h::Vector{Float64}        # holding cost in state 2 (failed)(length C)
    c_repair::Vector{Float64} # marginal repair cost            (length C)
    π::Float64                # setup cost (paid once per period with ≥1 repair)
end

# ============================================================================
# SURROGATE VALUES — Option B (paid-repair baseline)
# ============================================================================

"""
u[i, s, m] = cost-to-go for component i in state s with m steps remaining,
             under the single-asset paid-repair surrogate (Option B).

Each step: u[i,s,m] = min(wait cost, paid-repair cost)
  wait:   ℓ(s) + E[u_{m-1}(X^wait) | s]
  repair: (π + c_repair[i]) + E[u_{m-1}(X^rep) | s]

where E[X^rep | s] uses p_repair[i]: succeed → state 1, fail → natural transition.
State 2 is absorbing under no-repair.
"""
function surrogate_values(p::SetupCostParams2S)
    u = zeros(p.C, 2, p.H + 1)

    for i in 1:p.C
        π_i = p.π + p.c_repair[i]
        for m in 2:p.H+1
            E_wait = [
                (1 - p.p_fail[i]) * u[i, 1, m-1] + p.p_fail[i] * u[i, 2, m-1],
                u[i, 2, m-1]   # state 2 absorbing under no-repair
            ]
            E_rep = [p.p_repair[i] * u[i, 1, m-1] + (1 - p.p_repair[i]) * E_wait[s]
                     for s in 1:2]

            for s in 1:2
                hold       = s == 2 ? p.h[i] : 0.0
                u[i, s, m] = min(hold + E_wait[s],
                                 π_i  + E_rep[s])
            end
        end
    end
    return u
end

# ============================================================================
# REPAIR GAINS
# ============================================================================

"""
G[i,s,m]    = gross gain from repairing component i in state s with m steps left.
Gnet[i,s,m] = G[i,s,m] − c_repair[i]  (net gain after marginal repair cost).

G[i,s,m] = ℓ(s) + E[u_{m-1}(X^wait)|s] − E[u_{m-1}(X^rep)|s]
"""
function repair_gains(p::SetupCostParams2S, u::Array{Float64,3})
    G    = zeros(p.C, 2, p.H + 1)
    Gnet = zeros(p.C, 2, p.H + 1)

    for i in 1:p.C, m in 2:p.H+1
        E_wait = [
            (1 - p.p_fail[i]) * u[i, 1, m-1] + p.p_fail[i] * u[i, 2, m-1],
            u[i, 2, m-1]
        ]
        E_rep = [p.p_repair[i] * u[i, 1, m-1] + (1 - p.p_repair[i]) * E_wait[s]
                 for s in 1:2]

        for s in 1:2
            hold          = s == 2 ? p.h[i] : 0.0
            G[i, s, m]    = hold + E_wait[s] - E_rep[s]
            Gnet[i, s, m] = G[i, s, m] - p.c_repair[i]
        end
    end
    return G, Gnet
end

# ============================================================================
# HEURISTIC 1 — Net gains vs setup cost
# ============================================================================
#
# Without a budget constraint, the shadow price of opening a window is exactly
# the setup cost π. The decision rule is simply:
#
#   Open a window (repair all i with Gnet > 0)  iff  Σ_i [Gnet_i,m(x_i)]⁺ > π
#
# No λ* calibration is needed.

function simulate_h1(p::SetupCostParams2S, Gnet::Array{Float64,3}; seed=1234)
    Random.seed!(seed)
    total_loss = 0.0
    total_cost = 0.0
    x = fill(1, p.C)

    for m in p.H:-1:1
        Sm = sum(max(Gnet[i, x[i], m], 0.0) for i in 1:p.C)

        # holding cost at the START of the period (before action)
        total_loss += sum(p.h[i] * (x[i] == 2) for i in 1:p.C)

        if Sm > p.π
            R = [i for i in 1:p.C if Gnet[i, x[i], m] > 0]
            total_cost += p.π + sum(p.c_repair[i] for i in R)
            for i in R
                rand() < p.p_repair[i] && (x[i] = 1)
            end
        end

        # 2-state natural transitions (state 2 is absorbing without repair)
        for i in 1:p.C
            x[i] == 1 && rand() < p.p_fail[i] && (x[i] = 2)
        end
    end
    return total_loss + total_cost
end

# ============================================================================
# SHARED UTILITY — transition matrices
# ============================================================================

"""
Trans[i, s′, s, a] = P(next state = s′ | current state = s, action a).
No repair (a=1): 1→{1,2}, 2→{2} (absorbing).
Repair   (a=2): succeed → state 1 (prob p_repair); fail → natural transition.
"""
function build_trans(p::SetupCostParams2S)
    Trans = zeros(p.C, 2, 2, 2)
    for i in 1:p.C
        # no repair
        Trans[i, 1, 1, 1] = 1 - p.p_fail[i];  Trans[i, 2, 1, 1] = p.p_fail[i]
        Trans[i, 2, 2, 1] = 1.0

        # repair
        for s in 1:2, sp in 1:2
            Trans[i, sp, s, 2] = p.p_repair[i] * (sp == 1 ? 1.0 : 0.0) +
                                  (1 - p.p_repair[i]) * Trans[i, sp, s, 1]
        end
    end
    return Trans
end

# ============================================================================
# HEURISTIC 2 — Rolling-horizon LP
# ============================================================================

function rolling_lp(p::SetupCostParams2S, x_current::Vector{Int}, m::Int)
    m <= 0 && return nothing, nothing, 0.0
    Trans = build_trans(p)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # x[i, s, a, t]: no ww dimension — in the setup-cost problem there is no shared
    # window resource to track. w_t[t] simply captures whether any repair occurs
    # at period t (to charge the setup cost π once).
    @variable(model, x[1:p.C, 1:2, 1:2, 1:m] >= 0)
    @variable(model, 0 <= w_t[1:m] <= 1)

    # ── initial condition ─────────────────────────────────────────────────────
    for i in 1:p.C
        s_i = x_current[i]
        @constraint(model, sum(x[i, s_i, a, m] for a in 1:2) == 1.0)
        for s in 1:2
            s != s_i && @constraint(model, sum(x[i, s, a, m] for a in 1:2) == 0.0)
        end
    end

    # ── Kolmogorov flow ───────────────────────────────────────────────────────
    for t in 1:m-1, i in 1:p.C, sp in 1:2
        @constraint(model,
            sum(x[i, sp, a, t] for a in 1:2) ==
            sum(Trans[i, sp, s, a] * x[i, s, a, t+1] for s in 1:2, a in 1:2))
    end

    # ── setup trigger: w_t[t] ≥ prob component i repairs at t ────────────────
    for t in 1:m, i in 1:p.C
        @constraint(model, w_t[t] >= sum(x[i, s, 2, t] for s in 1:2))
    end

    # ── normalisation ─────────────────────────────────────────────────────────
    for t in 1:m, i in 1:p.C
        @constraint(model, sum(x[i, s, a, t] for s in 1:2, a in 1:2) == 1.0)
    end

    # ── objective ─────────────────────────────────────────────────────────────
    holding_cost = sum(p.h[i] * x[i, 2, a, t]
                       for i in 1:p.C, a in 1:2, t in 1:m)
    repair_cost  = sum(p.c_repair[i] * x[i, s, 2, t]
                       for i in 1:p.C, s in 1:2, t in 1:m)
    setup_cost   = p.π * sum(w_t[t] for t in 1:m)

    @objective(model, Min, holding_cost + repair_cost + setup_cost)

    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL && status != MOI.FEASIBLE_POINT
        @warn "LP(m=$m) failed: $status"
        return nothing, nothing, NaN
    end
    return value.(x), value.(w_t), objective_value(model)
end

function simulate_h2(p::SetupCostParams2S; seed=1234)
    Random.seed!(seed)
    total_loss = 0.0
    total_cost = 0.0
    x = fill(1, p.C)

    for m in p.H:-1:1
        x_opt, _, _ = rolling_lp(p, x, m)

        R = Int[]
        if x_opt !== nothing
            for i in 1:p.C
                s_i = x[i]
                x_opt[i, s_i, 2, m] > 0.5 && push!(R, i)
            end
        end

        # holding cost at the START of the period (before action)
        total_loss += sum(p.h[i] * (x[i] == 2) for i in 1:p.C)

        if !isempty(R)
            total_cost += p.π + sum(p.c_repair[i] for i in R)
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
# LP LOWER BOUND  (full horizon, all components in state 1)
# ============================================================================

function lp_lower_bound(p::SetupCostParams2S)
    _, _, lb = rolling_lp(p, fill(1, p.C), p.H)
    return lb
end

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

function run_experiment(; seed_params=42, nsim=500)
    println("\n" * "="^80)
    println("SETUP-COST MAINTENANCE — 2 STATES (good / failed)")
    println("No window budget: pay π whenever ≥1 component is maintained.")
    println("="^80)

    rng = Random.MersenneTwister(seed_params)
    C, H = 5, 24

    p = SetupCostParams2S(
        C        = C,
        H        = H,
        p_fail   = rand(rng, C) .* 0.25 .+ 0.05,   # U[0.05, 0.30]  1→2
        p_repair = fill(1.0, C),
        h        = rand(rng, C) .* 2.00 .+ 0.50,   # U[0.50, 2.50]
        c_repair = rand(rng, C) .* 0.30 .+ 0.05,   # U[0.05, 0.35]
        π        = 5.0
    )

    println("\nGlobal: C=$C  H=$H  π=$(p.π)")
    println("\n  Comp │ p_fail │ p_repair │   h    │ c_repair")
    println("  ─────┼────────┼──────────┼────────┼─────────")
    for i in 1:C
        @printf("    %d  │  %.3f │  %.3f   │  %.3f │  %.3f\n",
                i, p.p_fail[i], p.p_repair[i], p.h[i], p.c_repair[i])
    end

    # ── LP LOWER BOUND ────────────────────────────────────────────────────────
    println("\n" * "-"^80)
    println("LP LOWER BOUND (full horizon, all-healthy start)")
    println("-"^80)
    lb = lp_lower_bound(p)
    println("  LP (setup cost)  : ", round(lb, digits=4))

    # ── HEURISTIC 1 ───────────────────────────────────────────────────────────
    println("\n" * "-"^80)
    println("HEURISTIC 1: NET GAINS vs SETUP COST")
    println("  Rule: open window iff Σ_i [Gnet_i,m(x_i)]⁺ > π  (threshold = π, no calibration)")
    println("-"^80)
    u = surrogate_values(p)
    _, Gnet = repair_gains(p, u)
    costs_h1 = [simulate_h1(p, Gnet; seed=s) for s in 1:nsim]
    println("Results ($nsim sims):  mean=$(round(mean(costs_h1),digits=4))  std=$(round(std(costs_h1),digits=4))")

    # ── HEURISTIC 2 ───────────────────────────────────────────────────────────
    
    println("\n" * "-"^80)
    println("HEURISTIC 2: ROLLING HORIZON LP")
    println("-"^80)
    costs_h2 = [simulate_h2(p; seed=s) for s in 1:nsim]
    println("Results ($nsim sims):  mean=$(round(mean(costs_h2),digits=4))  std=$(round(std(costs_h2),digits=4))")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    println("\n" * "="^80)
    println("SUMMARY")
    println("="^80)
    @printf("  %-30s  %8s  %10s\n", "Method", "Mean", "Gap vs LB")
    println("  " * "─"^52)
    @printf("  %-30s  %8.4f\n", "LP lower bound", lb)
    for (label, costs) in [("H1 (net gains vs π)", costs_h1),
                            ("H2 (rolling LP)    ", costs_h2)]
        gap = (mean(costs) / lb - 1) * 100
        @printf("  %-30s  %8.4f  %+6.1f%%\n", label, mean(costs), gap)
    end
    println("="^80)
end

run_experiment()

end # module SetupCostHetero2S
