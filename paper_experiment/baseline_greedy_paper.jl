# baseline_greedy_paper.jl
#
# Two-step heuristic on the model of paper.tex (true cost Tc, quota of penalty-free days):
#
#   Offline, component by component, ignoring the quotas: for every component m alone, the
#     threshold θ_m ("maintain iff n_m ≥ θ_m") of least expected cost over the horizon of the
#     simulation, starting new. Cost of a period for m alone, with the true cost of the paper
#     restricted to m (no free day):
#       not maintained in state n: Σ_τ π_τ ℓ_m(n),
#       maintained in state n, starting on the earliest day F = FIRST_START_DAY = L + 1:
#         Rc_m + mobilization cost + Σ_{τ < F} π_τ ℓ_m(n) + γ Σ_{τ ∈ repair} π_τ + Σ_{τ after repair} π_τ ℓ_m(1).
#     γ ∈ [0, 1] (paragraph "Two-step baseline" of paper.tex) is the fraction of the penalty of the
#     maintenance days charged to m alone: several components are usually maintained together and
#     share the same maintenance days. γ = 1 charges them all to m.
#   Online: the components to maintain are S = {m : x_m ≥ θ_m} (if S does not fit in the
#     technician capacity, the components are added by decreasing priority x_m − θ_m, then
#     ℓ_m(x_m), as long as the set still fits). The penalty-free days are then given greedily so
#     as not to pay penalties, as far as the remaining quota allows: on every pole, the κ days of
#     largest loss π_τ Cl^s_τ are declared free, κ being the smallest multiple of QUOTA_STEP that
#     covers every day with a positive loss, capped by the remaining quota k (largest multiple of
#     QUOTA_STEP ≤ k). Among the schedules of S, the one of least true cost with these free days
#     is applied.
#
# Uses the enumeration of the schedules of fluid_enum_paper.jl (requires fluid_lp_cg.jl,
# fluid_cg_paper.jl and fluid_enum_paper.jl).

using Printf

"""
    optimize_thresholds_paper(fd; gamma = 1.0)

For every component m alone, the threshold θ_m ∈ {1, …, n_m + 1} (n_m + 1: never maintain) of
least expected cost over the horizon of the simulation, starting in state 1, without quota, with
the fraction `gamma` of the penalty of the maintenance days charged to m.
Returns `(θ, expected_costs)`; ties go to the smallest θ.
"""
function optimize_thresholds_paper(fd::FluidData; gamma::Float64 = 1.0)
    0.0 <= gamma <= 1.0 || error("gamma must be in [0, 1], got $gamma")
    T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
    T = fd.inst.T_period; F = FIRST_START_DAY
    θ = zeros(Int, fd.M); best_costs = zeros(fd.M)
    for m in 1:fd.M
        n = fd.n_states[m]; pen = fd.inst.poles[fd.pole_of[m]].daily_penalty
        ℓ(s) = capacity_loss(fd, m, s)
        cost_keep = [sum(pen) * ℓ(s) for s in 1:n]
        cost_maint = [fd.cc[m] + MOBILIZATION_COST + sum(pen[1:(F - 1)]) * ℓ(s) +
                      gamma * sum(pen[F:(F + fd.d[m] - 1)]) + sum(pen[(F + fd.d[m]):T]) * ℓ(1) for s in 1:n]
        best = Inf
        for θm in 1:(n + 1)
            Q = zeros(n, n); c = zeros(n)   # Q[to, from]
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

"Greedy free days of pole p: the κ days of largest loss w, κ the smallest multiple of QUOTA_STEP covering the days of positive loss, capped by k."
function greedy_free_days(w::Vector{Float64}, k::Int)
    n_pos = count(>(1e-12), w)
    κ = min(QUOTA_STEP * cld(n_pos, QUOTA_STEP), QUOTA_STEP * div(k, QUOTA_STEP))
    return best_free_days(w, κ)
end

"Greedy free days of every pole for the schedule `start` in the state (x, k), and the resulting joint action."
function greedy_column(fd::FluidData, x::Vector{Int}, k::Vector{Int}, start::Vector{Int})
    return Column(copy(start), [greedy_free_days(loss_weights(fd, x, start, p), k[p]) for p in 1:fd.inst.n_poles])
end

"""
Policy (x, k, t_global) -> (Column, solve_time) of the greedy two-step heuristic. The thresholds
are optimised once, at construction.
"""
function make_policy_baseline_greedy_paper(fd::FluidData; log_solve::Function = info -> nothing,
                                           gamma::Float64 = 1.0)
    θ, _ = optimize_thresholds_paper(fd; gamma = gamma)
    @printf("  baseline_greedy thresholds θ_m (maintain iff state >= θ_m), gamma = %.2f: %s\n", gamma, string(θ))
    E = joint_action_enum(fd)
    set_id = Dict(s => i for (i, s) in enumerate(E.sets))
    scheds_of = [Int[] for _ in E.sets]
    for (i, si) in enumerate(E.sched_set); push!(scheds_of[si], i); end

    function policy(x::Vector{Int}, k::Vector{Int}, t_global::Int)
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
            c = sum(true_period_cost(fd, x, col))
            c < best_cost - 1e-12 && (best = col; best_cost = c)
        end
        elapsed = time() - t_begin
        t0_cyc = (t_global - 1) % N_PERIODS_PER_YEAR + 1
        log_solve((key = (Tuple(x), Tuple(k), t0_cyc, 1), time = elapsed, lp_value = best_cost,
                   iterations = 0, n_columns = 0, converged = true, t_milp = 0.0, t_lp = 0.0))
        return best, elapsed
    end
    return policy
end
