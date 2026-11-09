module WildemanGroupingHetero

using Random, Statistics, Printf
using JuMP, HiGHS
using Base: @kwdef

# General n-state degradation model (n_states ≥ 2):
#
#   States: 1 = good, 2 = slightly degraded, …, n_states = panne (absorbing)
#   Transitions (no repair): state k → k+1  w.p. p_trans[i, k]   (k = 1..n_states-1)
#                            state n_states is absorbing
#   Repair: → state 1  w.p. p_repair[i],  else no-repair transition
#   Holding costs: h[i, k] for component i in state k  (h[i,1] = 0 by convention)
#
# Problem: no free / paid window budget.  Each period the planner chooses
# which components to repair.  Whenever ≥1 component is repaired a fixed
# setup cost π is charged (once per period).  Each repaired component also
# incurs its own marginal cost c_repair[i].
#
# This file compares three heuristics on the same problem instance:
#   H1 — Net-gains vs setup cost   (from setup_cost_hetero.jl)
#   H2 — Rolling-horizon fluid LP  (from setup_cost_hetero.jl)
#   H3 — Wildeman-Dekker grouping  (adapted from EJOR 1997)

@kwdef struct SetupCostParams
    C::Int
    H::Int
    n_states::Int                       # number of states (≥ 2)
    p_trans::Matrix{Float64}            # C × (n_states-1): P(k → k+1 | no repair)
    p_repair::Vector{Float64}           # C: P(repair → state 1)
    h::Matrix{Float64}                  # C × n_states: holding cost in each state
    c_repair::Vector{Float64}           # C: marginal repair cost
    π::Float64                          # setup cost (once per period with ≥1 repair)
end

# Holding cost for component i in state s
hcost(p::SetupCostParams, i::Int, s::Int) = p.h[i, s]

# One-step stochastic transition (in-place, works for any n_states)
function transition!(x_state::Vector{Int}, p::SetupCostParams)
    for i in 1:p.C
        s = x_state[i]
        s < p.n_states && rand() < p.p_trans[i, s] && (x_state[i] = s + 1)
    end
end

# ============================================================================
# SHARED UTILITY — transition matrices
# ============================================================================

function build_trans(p::SetupCostParams)
    S     = p.n_states
    Trans = zeros(p.C, S, S, 2)
    for i in 1:p.C
        # No-repair: state k → k+1 w.p. p_trans[i,k]; last state absorbing
        for k in 1:S-1
            Trans[i, k,   k, 1] = 1 - p.p_trans[i, k]
            Trans[i, k+1, k, 1] = p.p_trans[i, k]
        end
        Trans[i, S, S, 1] = 1.0
        # With repair: → state 1 w.p. p_repair, else no-repair transition
        for s in 1:S, sp in 1:S
            Trans[i, sp, s, 2] = p.p_repair[i] * (sp == 1 ? 1.0 : 0.0) +
                                  (1 - p.p_repair[i]) * Trans[i, sp, s, 1]
        end
    end
    return Trans
end

# ============================================================================
# HEURISTIC 1 — Net gains vs setup cost  (identical to setup_cost_hetero.jl)
# ============================================================================

"""
u[i, s, m] = cost-to-go for component i in state s with m steps remaining,
             under the single-asset paid-repair surrogate (Option B baseline).
"""
function surrogate_values(p::SetupCostParams)
    S     = p.n_states
    Trans = build_trans(p)
    u     = zeros(p.C, S, p.H + 1)
    for i in 1:p.C
        π_i = p.π + p.c_repair[i]
        for m in 2:p.H+1
            E_wait = [sum(Trans[i, sp, s, 1] * u[i, sp, m-1] for sp in 1:S) for s in 1:S]
            E_rep  = [sum(Trans[i, sp, s, 2] * u[i, sp, m-1] for sp in 1:S) for s in 1:S]
            for s in 1:S
                u[i, s, m] = min(hcost(p, i, s) + E_wait[s], π_i + E_rep[s])
            end
        end
    end
    return u
end

"""
Gnet[i, s, m] = net gain from repairing component i in state s with m steps left.
"""
function repair_gains(p::SetupCostParams, u::Array{Float64,3})
    S     = p.n_states
    Trans = build_trans(p)
    G     = zeros(p.C, S, p.H + 1)
    Gnet  = zeros(p.C, S, p.H + 1)
    for i in 1:p.C, m in 2:p.H+1
        E_wait = [sum(Trans[i, sp, s, 1] * u[i, sp, m-1] for sp in 1:S) for s in 1:S]
        E_rep  = [sum(Trans[i, sp, s, 2] * u[i, sp, m-1] for sp in 1:S) for s in 1:S]
        for s in 1:S
            G[i, s, m]    = hcost(p, i, s) + E_wait[s] - E_rep[s]
            Gnet[i, s, m] = G[i, s, m] - p.c_repair[i]
        end
    end
    return G, Gnet
end

"""
H1: open a window iff Σ_i [Gnet_i,m(x_i)]⁺ > π; repair all i with Gnet > 0.
No λ* calibration needed — the setup cost π plays the role of the shadow price.
"""
function simulate_h1(p::SetupCostParams, Gnet::Array{Float64,3}; seed=1234)
    Random.seed!(seed)
    total_loss = 0.0
    total_cost = 0.0
    x = fill(1, p.C)

    for m in p.H:-1:1
        Sm = sum(max(Gnet[i, x[i], m], 0.0) for i in 1:p.C)

        total_loss += sum(hcost(p, i, x[i]) for i in 1:p.C)

        if Sm > p.π
            R = [i for i in 1:p.C if Gnet[i, x[i], m] > 0]
            total_cost += p.π + sum(p.c_repair[i] for i in R)
            for i in R
                rand() < p.p_repair[i] && (x[i] = 1)
            end
        end

        transition!(x, p)
    end
    return total_loss + total_cost
end

# ============================================================================
# HEURISTIC 2 — Rolling-horizon fluid LP  (identical to setup_cost_hetero.jl)
# ============================================================================

function rolling_lp(p::SetupCostParams, x_current::Vector{Int}, m::Int)
    m <= 0 && return nothing, nothing, 0.0
    S     = p.n_states
    Trans = build_trans(p)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[1:p.C, 1:S, 1:2, 1:m] >= 0)
    @variable(model, 0 <= w_t[1:m] <= 1)

    for i in 1:p.C
        s_i = x_current[i]
        @constraint(model, sum(x[i, s_i, a, m] for a in 1:2) == 1.0)
        for s in 1:S
            s != s_i && @constraint(model, sum(x[i, s, a, m] for a in 1:2) == 0.0)
        end
    end

    for t in 1:m-1, i in 1:p.C, sp in 1:S
        @constraint(model,
            sum(x[i, sp, a, t] for a in 1:2) ==
            sum(Trans[i, sp, s, a] * x[i, s, a, t+1] for s in 1:S, a in 1:2))
    end

    for t in 1:m, i in 1:p.C
        @constraint(model, w_t[t] >= sum(x[i, s, 2, t] for s in 1:S))
    end

    for t in 1:m, i in 1:p.C
        @constraint(model, sum(x[i, s, a, t] for s in 1:S, a in 1:2) == 1.0)
    end

    holding_cost = sum(hcost(p, i, s) * x[i, s, a, t]
                       for i in 1:p.C, s in 1:S, a in 1:2, t in 1:m)
    repair_cost  = sum(p.c_repair[i] * x[i, s, 2, t]
                       for i in 1:p.C, s in 1:S, t in 1:m)
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

function simulate_h2(p::SetupCostParams; seed=1234)
    Random.seed!(seed)
    total_loss = 0.0
    total_cost = 0.0
    x = fill(1, p.C)

    for m in p.H:-1:1
        x_opt, _, _ = rolling_lp(p, x, m)

        R = Int[]
        if x_opt !== nothing
            for i in 1:p.C
                x_opt[i, x[i], 2, m] > 0.5 && push!(R, i)
            end
        end

        total_loss += sum(hcost(p, i, x[i]) for i in 1:p.C)

        if !isempty(R)
            total_cost += p.π + sum(p.c_repair[i] for i in R)
            for i in R
                rand() < p.p_repair[i] && (x[i] = 1)
            end
        end

        transition!(x, p)
    end
    return total_loss + total_cost
end

lp_lower_bound(p::SetupCostParams) = rolling_lp(p, fill(1, p.C), p.H)[3]

# ============================================================================
# HEURISTIC 3 — Wildeman-Dekker exact dynamic grouping  (EJOR 1997)
# ============================================================================
#
# Exact adaptation of Wildeman & Dekker (1997) to discrete-time Markov
# deterioration.  Four steps follow the paper directly:
#
# Step 1 — Individual optimization (§2 of the paper)
#   φ_i(x) = [π + c_repair_i + E_hold_i(x)] / x,   x = 1…H
#   E_hold_i(x) = Σ_{t=1}^{x} E[h_i(s_t) | s_0=1]  (Markov forward iteration)
#   x*_i = argmin_x φ_i(x),   φ*_i = φ_i(x*_i)
#
# Step 2 — Penalty cost (§3)
#   h_i(Δt) = Γ_i(x*+Δt) − Γ_i(x*) − Δt·φ*_i  where Γ_i(x) = c_repair_i + E_hold_i(x)
#   c_repair cancels:  h_i(Δt) = E_hold_i(x*+Δt) − E_hold_i(x*) − Δt·φ*_i
#   Admissible interval I_n = { Δt : h_n(Δt) ≤ π }  (penalty ≤ savings from one setup)
#
# Step 3 — Optimal group times & savings (§4)
#   Group G = {j..k} (consecutive by tentative time) is admissible iff ∩ I_n ≠ ∅.
#   t_G = argmin_{t ∈ ∩ I_n} Σ h_n(t − t_n);   Q*_G = (|G|−1)·π − Σ h_n(t_G−t_n)
#
# Step 4 — Backtracking DP (§5)
#   dp[k] = max total savings for operations 1..k
#   dp[k] = max_{j≤k} dp[j-1] + Q*_{j..k}
#
# Rolling-horizon adaptation: re-plan after every actual repair event.

"""
hold[i, x] = Σ_{t=1}^{x} E[h_deg_i·1{s_t=2} + h_loss_i·1{s_t=3} | s_0=1]
Cumulative expected holding cost over a cycle of length x from state 1.
"""
function precompute_hold(p::SetupCostParams)
    S    = p.n_states
    hold = zeros(p.C, p.H)
    for i in 1:p.C
        prob = zeros(S);  prob[1] = 1.0
        for x in 1:p.H
            new_prob = zeros(S)
            for k in 1:S-1
                new_prob[k]   += prob[k] * (1 - p.p_trans[i, k])
                new_prob[k+1] += prob[k] * p.p_trans[i, k]
            end
            new_prob[S] += prob[S]
            prob = new_prob
            hold[i, x] = (x > 1 ? hold[i, x-1] : 0.0) +
                          sum(p.h[i, s] * prob[s] for s in 1:S)
        end
    end
    return hold
end

# φ_i(x) = (π + c_repair_i + E_hold_i(x)) / x
avg_cost(p::SetupCostParams, i::Int, x::Int, hold::Matrix{Float64}) =
    (0 < x <= p.H) ? (p.π + p.c_repair[i] + hold[i, x]) / x : Inf

function individual_optimal(p::SetupCostParams, i::Int, hold::Matrix{Float64})
    best_x, best_phi = 1, avg_cost(p, i, 1, hold)
    for x in 2:p.H
        φ = avg_cost(p, i, x, hold)
        if φ < best_phi
            best_phi = φ;  best_x = x
        end
    end
    return best_x, best_phi
end

# h_i(Δt) = Γ_i(x*+Δt) − Γ_i(x*) − Δt·φ*   (c_repair cancels)
function penalty_cost(p::SetupCostParams, i::Int, x_star::Int, phi_star::Float64,
                      delta::Int, hold::Matrix{Float64})
    x_new = x_star + delta
    (x_new <= 0 || x_new > p.H) && return Inf
    return hold[i, x_new] - hold[i, x_star] - delta * phi_star
end

"""
    wildeman_plan(p, x_stars, phi_stars, hold, last_rep, from_period)

Implements the exact Wildeman-Dekker (1997) algorithm:

1. Generate all tentative operations (i, t_n) with t_n = last_rep[i] + j·x*[i].
2. For each operation n find the admissible shift interval [dm[n], dp[n]] such that
   h_n(Δt) ≤ π (penalty ≤ setup cost saved by grouping).
3. For every consecutive sub-sequence {j..k} whose admissible intervals share a
   non-empty intersection, find t_G minimising Σ h_n(t − t_n) and compute
   Q*_{j..k} = (k−j)·π − Σ h_n(t_G − t_n).
4. DP over operations: dp[k] = max total savings for ops 1..k;
   transition dp[k] = max_{j≤k} dp[j-1] + Q*_{j..k}.
5. Backtrack to recover the optimal partition into groups and their execution times.

Returns a Dict{Int,Vector{Int}}: period → list of component indices to repair.
"""
function wildeman_plan(p::SetupCostParams, x_stars::Vector{Int},
                       phi_stars::Vector{Float64}, hold::Matrix{Float64},
                       last_rep::Vector{Int}, from_period::Int)

    # ── 1. Tentative operations ───────────────────────────────────────────────
    ops = Tuple{Int,Int}[]          # (component_index, tentative_time)
    for i in 1:p.C
        t = last_rep[i] + x_stars[i]
        while t <= p.H
            push!(ops, (i, t))
            t += x_stars[i]
        end
    end
    isempty(ops) && return Dict{Int,Vector{Int}}()
    sort!(ops, by = o -> o[2])
    N = length(ops)

    # ── 2. Admissible shift intervals [dm[n], dp[n]] where h_n(Δt) ≤ π ──────
    dm    = zeros(Int, N)           # most-negative admissible shift (≤ 0)
    dplus = zeros(Int, N)           # most-positive admissible shift (≥ 0)
    for n in 1:N
        i = ops[n][1]
        Δ = -1
        while x_stars[i] + Δ > 0
            penalty_cost(p, i, x_stars[i], phi_stars[i], Δ, hold) <= p.π || break
            dm[n] = Δ;  Δ -= 1
        end
        Δ = 1
        while x_stars[i] + Δ <= p.H
            penalty_cost(p, i, x_stars[i], phi_stars[i], Δ, hold) <= p.π || break
            dplus[n] = Δ;  Δ += 1
        end
    end

    # ── 3. Precompute penalty table: pen_tab[n, t] = h_n(t − t_n) ───────────
    # Clamp to a large finite value so sums never become Inf/NaN.
    pen_tab = fill(1e12, N, p.H)
    for n in 1:N
        i, t_n = ops[n]
        for t in from_period:p.H
            δ  = t - t_n
            xn = x_stars[i] + δ
            if 0 < xn <= p.H
                pen_tab[n, t] = hold[i, xn] - hold[i, x_stars[i]] - δ * phi_stars[i]
            end
        end
    end

    # ── 4. Savings matrix: savings[j,k] = Q*_{j..k}, gtime[j,k] = t_G ───────
    savings = fill(-Inf, N, N)
    gtime   = zeros(Int, N, N)

    for j in 1:N
        # Effective admissible window for operation j, clamped to [from_period, H]
        lo = max(from_period, ops[j][2] + dm[j])
        hi = min(p.H,         ops[j][2] + dplus[j])
        # If the whole admissible window is in the past, force execution now.
        hi < from_period && (lo = from_period; hi = from_period)

        for k in j:N
            if k > j
                lo = max(lo, max(from_period, ops[k][2] + dm[k]))
                hi = min(hi, min(p.H,         ops[k][2] + dplus[k]))
                lo > hi && break      # intersection empty → no larger group possible
            end

            # Find t_G ∈ [lo, hi] minimising Σ_n h_n(t − t_n)
            best_t = lo
            best_c = sum(pen_tab[n, lo] for n in j:k)
            for t in lo+1:hi
                c = sum(pen_tab[n, t] for n in j:k)
                if c < best_c;  best_c = c;  best_t = t;  end
            end
            savings[j, k] = (k - j) * p.π - best_c
            gtime[j, k]   = best_t
        end
    end

    # ── 5. DP ─────────────────────────────────────────────────────────────────
    # dpv[k+1] = max savings using operations 1..k  (dpv[1] = 0 for empty prefix)
    dpv  = fill(-Inf, N + 1);  dpv[1] = 0.0
    back = zeros(Int, N + 1)   # back[k+1] = j: ops j..k form the last group

    for k in 1:N
        for j in 1:k
            savings[j, k] == -Inf && continue
            val = dpv[j] + savings[j, k]
            if val > dpv[k + 1]
                dpv[k + 1] = val
                back[k + 1] = j
            end
        end
    end

    # ── 6. Backtrack → scheduled groups ──────────────────────────────────────
    sched = Dict{Int,Vector{Int}}()
    k = N
    while k >= 1
        j = back[k + 1]
        j == 0 && break                     # should not happen for valid instances
        t_G = gtime[j, k]
        for n in j:k
            push!(get!(sched, t_G, Int[]), ops[n][1])
        end
        k = j - 1
    end
    return sched
end

"""
    simulate_h3(p; seed, corrective)

Simulate the Wildeman-Dekker heuristic.

`corrective = false` (paper-faithful):
  Only execute pre-planned groups.  Components in state 3 accumulate holding
  costs but are NOT forced into an unplanned repair; the plan is never altered
  by actual state realisations.  This matches the paper's deterministic setting.

`corrective = true` (stochastic extension):
  When any component reaches state 3, a corrective repair is forced.  The
  setup already triggered is then used as an opportunity window: every other
  component j is included if h_j(Δt_j) ≤ π (Wildeman criterion).  The full
  plan is recomputed after every repair event.
"""
function simulate_h3(p::SetupCostParams; seed=1234, corrective::Bool=true)
    Random.seed!(seed)
    hold      = precompute_hold(p)
    x_stars   = zeros(Int, p.C)
    phi_stars = zeros(Float64, p.C)
    for i in 1:p.C
        x_stars[i], phi_stars[i] = individual_optimal(p, i, hold)
    end

    x_state    = fill(1, p.C)
    last_rep   = fill(0, p.C)
    total_loss = 0.0
    total_cost = 0.0

    for period in 1:p.H
        # ── Recompute optimal grouping at every period from current last_rep ──
        sched = wildeman_plan(p, x_stars, phi_stars, hold, last_rep, period)

        # Accumulate holding costs for the current state (before repair)
        total_loss += sum(hcost(p, i, x_state[i]) for i in 1:p.C)

        # Start from the group prescribed by the fresh plan for this period.
        # unique! guards against the same component appearing twice in one group.
        to_repair = unique!(copy(get(sched, period, Int[])))

        if corrective
            # ── Corrective: force repair of any component in the broken state ──
            for i in 1:p.C
                x_state[i] == p.n_states && i ∉ to_repair && push!(to_repair, i)
            end

            # ── Opportunistic: if a setup is triggered, apply Wildeman criterion
            #    to every remaining component j: include if h_j(Δt_j) ≤ π ──────
            if !isempty(to_repair)
                for i in 1:p.C
                    i ∈ to_repair && continue
                    t_i = last_rep[i] + x_stars[i]
                    Δ   = period - t_i
                    pen = penalty_cost(p, i, x_stars[i], phi_stars[i], Δ, hold)
                    pen <= p.π && push!(to_repair, i)
                end
            end
        end

        if !isempty(to_repair)
            total_cost += p.π + sum(p.c_repair[i] for i in to_repair)
            for i in to_repair
                rand() < p.p_repair[i] && (x_state[i] = 1)
                last_rep[i] = period
            end
            # No post-repair replan: the next period's iteration will replan
            # from the updated last_rep automatically.
        end

        # Stochastic state transitions
        transition!(x_state, p)
    end
    return total_loss + total_cost
end

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

function run_experiment(; seed_params=42, nsim=500, n_states=3)
    println("\n" * "="^80)
    label = n_states == 3 ? "3 STATES (good / degraded / panne)" : "2 STATES (good / panne)"
    println("SETUP-COST MAINTENANCE — $label")
    println("Comparing H1 (net gains), H3a/H3b (Wildeman grouping)")
    println("="^80)

    rng = Random.MersenneTwister(seed_params)
    C, H = 5, 24
    S    = n_states

    # ── Transition probabilities: p_trans[i,k] = P(state k → k+1)
    # First transition (1→2) is slow; later transitions are progressively faster.
    p_trans = zeros(C, S - 1)
    for k in 1:S-1
        lo = min(0.05 * k, 0.40)
        hi = min(0.05 * k + 0.20, 0.70)
        p_trans[:, k] = rand(rng, C) .* (hi - lo) .+ lo
    end

    # ── Holding costs: 0 in state 1; increasing toward last state (panne)
    h = zeros(C, S)
    h_max = rand(rng, C) .* 2.00 .+ 0.50    # cost in panne state
    h[:, S] = h_max
    for k in 2:S-1
        frac = (k - 1) / (S - 1)            # linearly spaced fractions 0→1
        h[:, k] = h_max .* frac .* (rand(rng, C) .* 0.4 .+ 0.8)
    end

    p = SetupCostParams(
        C        = C,
        H        = H,
        n_states = S,
        p_trans  = p_trans,
        p_repair = fill(1.0, C),
        h        = h,
        c_repair = rand(rng, C) .* 0.30 .+ 0.05,
        π        = 5.0
    )

    println("\nGlobal: C=$C  H=$H  n_states=$S  π=$(p.π)")

    # Transition probabilities table
    header = "  Comp │ p_repair │ c_repair │" * join([" p($(k)→$(k+1)) │" for k in 1:S-1])
    println("\n" * header)
    println("  " * "─"^(length(header)-2))
    for i in 1:C
        row = @sprintf("    %d  │  %.3f   │  %.3f   │", i, p.p_repair[i], p.c_repair[i])
        for k in 1:S-1
            row *= @sprintf("   %.3f   │", p.p_trans[i, k])
        end
        println(row)
    end

    # Holding costs table
    println("\n  Comp │" * join([" h(state $k) │" for k in 2:S]))
    println("  " * "─"^(10 + 13*(S-1)))
    for i in 1:C
        row = @sprintf("    %d  │", i)
        for k in 2:S
            row *= @sprintf("    %.3f    │", p.h[i, k])
        end
        println(row)
    end

    # ── Wildeman individual optima ────────────────────────────────────────────
    hold = precompute_hold(p)
    println("\n  Wildeman individual optimal intervals (stand-alone):")
    println("  Comp │  x*  │  φ* (avg cost / period)")
    println("  ─────┼──────┼────────────────────────")
    for i in 1:C
        xs, phis = individual_optimal(p, i, hold)
        @printf("    %d  │  %2d  │  %.4f\n", i, xs, phis)
    end

    # ── LP lower bound ────────────────────────────────────────────────────────
    println("\n" * "-"^80)
    println("LP LOWER BOUND (full horizon, all-healthy start)")
    println("-"^80)
    lb = lp_lower_bound(p)
    println("  LP (fluid relaxation): ", round(lb, digits=4))

    # ── H3a — paper-faithful (no corrective) ─────────────────────────────────
    println("\n" * "-"^80)
    println("HEURISTIC 3a: WILDEMAN-DEKKER — paper-faithful (corrective=false)")
    println("  Only pre-planned groups are executed.  Broken components are NOT")
    println("  forced into unplanned repairs; they keep accumulating holding costs.")
    println("-"^80)
    costs_h3a = [simulate_h3(p; seed=s, corrective=false) for s in 1:nsim]
    @printf("Results (%d sims):  mean=%.4f  std=%.4f  gap=%+.1f%%\n",
            nsim, mean(costs_h3a), std(costs_h3a), (mean(costs_h3a)/lb - 1)*100)

    # ── H3b — stochastic extension (corrective + opportunistic) ───────────────
    println("\n" * "-"^80)
    println("HEURISTIC 3b: WILDEMAN-DEKKER — stochastic extension (corrective=true)")
    println("  State-3 forces repair; setup used as opportunity window for others")
    println("  via Wildeman criterion h_j(Δt_j) ≤ π.  Replans after every repair.")
    println("-"^80)
    costs_h3b = [simulate_h3(p; seed=s, corrective=true) for s in 1:nsim]
    @printf("Results (%d sims):  mean=%.4f  std=%.4f  gap=%+.1f%%\n",
            nsim, mean(costs_h3b), std(costs_h3b), (mean(costs_h3b)/lb - 1)*100)

    # ── H1 ────────────────────────────────────────────────────────────────────
    println("\n" * "-"^80)
    println("HEURISTIC 1: NET GAINS vs SETUP COST")
    println("  Rule: open window iff Σ_i [Gnet_i,m(x_i)]⁺ > π")
    println("-"^80)
    u = surrogate_values(p)
    _, Gnet = repair_gains(p, u)
    costs_h1 = [simulate_h1(p, Gnet; seed=s) for s in 1:nsim]
    @printf("Results (%d sims):  mean=%.4f  std=%.4f  gap=%+.1f%%\n",
            nsim, mean(costs_h1), std(costs_h1), (mean(costs_h1)/lb - 1)*100)


    #=
    # ── H2 ────────────────────────────────────────────────────────────────────
    println("\n" * "-"^80)
    println("HEURISTIC 2: ROLLING HORIZON FLUID LP")
    println("-"^80)
    costs_h2 = [simulate_h2(p; seed=s) for s in 1:nsim]
    @printf("Results (%d sims):  mean=%.4f  std=%.4f  gap=%+.1f%%\n",
            nsim, mean(costs_h2), std(costs_h2), (mean(costs_h2)/lb - 1)*100)


    # ── Summary ───────────────────────────────────────────────────────────────
    println("\n" * "="^80)
    println("SUMMARY")
    println("="^80)
    @printf("  %-40s  %8s  %10s\n", "Method", "Mean", "Gap vs LB")
    println("  " * "─"^62)
    @printf("  %-40s  %8.4f\n", "LP lower bound", lb)
    for (label, costs) in [
            ("H3a Wildeman — paper-faithful        ", costs_h3a),
            ("H3b Wildeman — corrective+opportun.  ", costs_h3b),
            ("H1  net gains vs π                   ", costs_h1)]
        gap = (mean(costs) / lb - 1) * 100
        @printf("  %-40s  %8.4f  %+6.1f%%\n", label, mean(costs), gap)
    end
    println("="^80)
    =#
end

#run_experiment(n_states=2)
#run_experiment(n_states=3)
run_experiment(n_states=5)

end # module WildemanGroupingHetero
