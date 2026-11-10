# two_step_baseline.jl
#
# 2SB of paper.tex: the two-step baseline (Section "Setup", paragraph "Two-step baseline").
#
#   Step 1 (offline). Every component m is considered alone, without quota. Its policy is optimised
#     among the threshold policies "maintain m iff n_m ≥ θ_m", θ_m ∈ {1, …, N_m + 1}, by minimising
#     its expected cost over the horizon from the new state. Cost of a period of m alone in state n:
#       not maintained: T π ℓ_m(n),
#       maintained, full model: c^rep_m + L π ℓ_m(n) + γ d_m π (the repair starts on day L + 1, and
#         the penalty after the repair is that of the new state, 0 for every component),
#       maintained, simplified model: T π ℓ_m(n) + c^rep_m + γ d_m π (the failure cost of the period is
#         paid whether the component is maintained or not).
#     γ ∈ [0, 1] is the fraction of the cost of the maintenance days charged to m alone, chosen by
#     scripts/tune_gamma.jl.
#   Step 2 (online). The components with n_m ≥ θ_m are due for maintenance. If they cannot all be
#     maintained within the technician capacity, they are considered by decreasing priority n_m − θ_m,
#     ties being broken by decreasing capacity loss, and every component is kept if it can be
#     maintained together with the components kept so far. In the full model, the declared days are
#     chosen greedily, so as to avoid as much penalty as possible in the current period, as long as the
#     remaining quota allows it. Among the schedules of the components kept, the one of least cost of
#     the period is applied.

using Printf

"""
    optimize_thresholds(fd; gamma = 1.0)

Step 1: for every component m alone, the threshold θ_m ∈ {1, …, N_m + 1} (N_m + 1: never maintain) of
least expected cost over the horizon, starting new. Returns `(θ, expected_costs)`; ties go to the
smallest θ.
"""
function optimize_thresholds(fd::FluidData; gamma::Float64 = 1.0)
    0.0 <= gamma <= 1.0 || error("gamma must be in [0, 1], got $gamma")
    T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
    T = fd.inst.T_period; F = FIRST_START_DAY
    θ = zeros(Int, fd.M); best_costs = zeros(fd.M)
    for m in 1:fd.M
        n = fd.n_states[m]; pen = fd.inst.poles[fd.pole_of[m]].daily_penalty
        ℓ(s) = capacity_loss(fd, m, s)
        repair_days = F:(F + fd.d[m] - 1)
        cost_keep = [sum(pen) * ℓ(s) for s in 1:n]
        cost_maint = fd.inst.simplified ?
            [sum(pen) * ℓ(s) + fd.cc[m] + gamma * sum(pen[repair_days]) for s in 1:n] :
            [fd.cc[m] + sum(pen[1:(F - 1)]) * ℓ(s) + gamma * sum(pen[repair_days]) + sum(pen[(F + fd.d[m]):T]) * ℓ(1) for s in 1:n]
        best = Inf
        for θm in 1:(n + 1)
            Q = zeros(n, n); c = zeros(n)   # transition matrix Q[to, from] and cost of the threshold policy θm
            for s in 1:n
                if s >= θm
                    Q[1, s] = 1.0; c[s] = cost_maint[s]
                else
                    Q[:, s] = fd.P[m][:, s]; c[s] = cost_keep[s]
                end
            end
            v = zeros(n); v[1] = 1.0; total = 0.0
            for _ in 1:T_total
                total += sum(v .* c); v = Q * v
            end
            if total < best - 1e-9
                best = total; θ[m] = θm
            end
        end
        best_costs[m] = best
    end
    return θ, best_costs
end

"Greedy declared days of substation p: the κ days of largest penalty w, κ the smallest multiple of STEP covering the days of positive penalty, capped by the remaining quota k."
function greedy_free_days(w::Vector{Float64}, k::Int)
    n_pos = count(>(1e-12), w)
    κ = min(STEP * cld(n_pos, STEP), STEP * div(k, STEP))
    return best_free_days(w, κ)
end

"Joint action of the schedule `start` in the state (x, k), with the greedy declared days of every substation."
greedy_column(fd::FluidData, x::Vector{Int}, k::Vector{Int}, start::Vector{Int}) =
    Column(copy(start), [greedy_free_days(period_weights(fd, x, start, p), k[p]) for p in 1:fd.inst.n_poles])

"""
Policy (x, k, t) -> (Column, computing time) of the two-step baseline. The thresholds are optimised
once, at construction (step 1).
"""
function make_policy_two_step_baseline(fd::FluidData; gamma::Float64 = 1.0, log_solve::Function = info -> nothing)
    θ, _ = optimize_thresholds(fd; gamma = gamma)
    @printf("  2SB thresholds θ_m (maintain iff state >= θ_m), gamma = %.2f: %s\n", gamma, string(θ))
    E = joint_action_enum(fd)
    set_id = Dict(s => i for (i, s) in enumerate(E.sets))     # sets of components that fit in the capacity
    scheds_of = [Int[] for _ in E.sets]
    for (i, si) in enumerate(E.sched_set); push!(scheds_of[si], i); end

    function policy(x::Vector{Int}, k::Vector{Int}, t::Int)
        t_begin = time()
        S = BitVector([x[m] >= θ[m] for m in 1:fd.M])
        if !haskey(set_id, S)   # S does not fit in the technician capacity: priority order
            order = sort([m for m in 1:fd.M if S[m]]; by = m -> (-(x[m] - θ[m]), -capacity_loss(fd, m, x[m])))
            S = falses(fd.M)
            for m in order
                S[m] = true
                haskey(set_id, S) || (S[m] = false)
            end
        end
        best = nothing; best_cost = Inf
        for i in scheds_of[set_id[S]]
            col = greedy_column(fd, x, k, E.scheds[i])
            c = sum(period_cost(fd, x, col))
            c < best_cost - 1e-12 && (best = col; best_cost = c)
        end
        elapsed = time() - t_begin
        t0_cyc = (t - 1) % N_PERIODS_PER_YEAR + 1
        log_solve((key = decision_key(x, k, t0_cyc, 1), time = elapsed, lp_value = best_cost, t_milp = 0.0, t_lp = 0.0))
        return best, elapsed
    end
    return policy
end
