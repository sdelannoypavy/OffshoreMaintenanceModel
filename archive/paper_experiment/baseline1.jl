# baseline1.jl
#
# Baseline 1, "two-step optimisation". Written for the SIMPLIFIED problem (Instance.simplified,
# i.e. no free-day quota, no within-period transition); on the original instances (with a
# quota), step 2 additionally declares free days, greedily, within the remaining quota:
#
#   Step 1 (offline, once per instance): optimise a maintenance policy for each component
#     separately, among threshold policies "maintain iff state >= θ_m". The single-component
#     cost accounts for the failure cost δ_m(x), the replacement cost cc_m and a maintenance
#     cost equal to gamma (γ, default 1) times the duration d_m of the maintenance (times the
#     daily penalty, 1 here): γ < 1 shares the maintenance days with the other components
#     usually maintained at the same time, so that not all of them are charged to m alone.
#     No technician constraint and no coupling between components or poles is seen here.
#   Step 2 (online, in the simulation): for the current joint state x, the components to
#     maintain are S = {m : x_m >= θ_m}; the schedule is the one that minimises the true
#     period cost (Σ_{m not maintained} δ_m + Σ_{m maintained} cc_m + f(b), f(b) = number of
#     days each pole is under maintenance) among the schedules maintaining (a subset of) S.
#     It is applied, then the procedure is repeated at the next period.
#
# Requires fluid_lp_cg.jl to be loaded first (FluidData, Column, empty_column, solve_pricing,
# stage_cyc, N_PERIODS_PER_YEAR, ...).

using Printf, JLD2

"""
    optimize_thresholds(fd; gamma = 1.0)

`gamma` (γ) scales the maintenance-days part of the maintenance cost of step 1 only (step 2
always uses the true costs).

For every component m, the threshold θ_m ∈ {1, …, n_m + 1} (θ = n_m + 1: never maintain)
minimising the expected total cost of m alone over the horizon of the simulation
(`inst.n_years * N_PERIODS_PER_YEAR` periods, starting in state 1, as in `simulate_scenario`).
Per period, in state x: δ_m(x) if x < θ_m (then x moves according to the period transition
matrix), cc_m + γ * d_m * (mean daily penalty) if x >= θ_m (then x moves to 1).
`gamma` must be in [0, 1] (checked). Returns `(θ, expected_costs)`; ties go to the smallest θ.
"""
function optimize_thresholds(fd::FluidData; gamma::Float64 = 1.0)
    0.0 <= gamma <= 1.0 || error("gamma must be in [0, 1], got $gamma")
    T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
    θ = zeros(Int, fd.M)
    best_costs = zeros(fd.M)
    for m in 1:fd.M
        n = fd.n_states[m]
        pen = sum(fd.inst.poles[fd.pole_of[m]].daily_penalty) / fd.inst.T_period
        maint_cost = fd.cc[m] + gamma * fd.d[m] * pen
        best_cost = Inf
        for θm in 1:(n + 1)
            Q = zeros(n, n)          # Q[to, from]
            c = zeros(n)
            for x in 1:n
                if x >= θm
                    Q[1, x] = 1.0; c[x] = fd.δ[m][x] + maint_cost
                else
                    Q[:, x] = fd.P[m][:, x]; c[x] = fd.δ[m][x]
                end
            end
            v = zeros(n); v[1] = 1.0
            total = 0.0
            for _ in 1:T_total
                total += sum(v .* c)
                v = Q * v
            end
            if total < best_cost - 1e-9
                best_cost = total; θ[m] = θm
            end
        end
        best_costs[m] = best_cost
    end
    return θ, best_costs
end

"""
    schedule_min_cost(fd, x, S, θ)

Schedule maintaining the components `S::Vector{Bool}` chosen by the thresholds `θ`, with the
smallest period cost. All of S is maintained if a feasible schedule exists. Otherwise (the
technician-capacity constraint (11) is shared), the components are considered by decreasing
priority x_m - θ_m (ties: decreasing δ_m(x_m)) and each one is kept if the set kept so far plus
this component still admits a feasible schedule, else it is left unmaintained this period.
Feasibility and the final schedule are computed with the pricing MILP of fluid_lp_cg.jl (same
start-day grid, lead margin and constraint (11)), in which the maintenance of the wanted set
is forced (a large negative cost on "maintained", `FORCE`, a huge cost for m outside the set),
so that its objective is then the number of maintenance days f(b) (plus the replacement costs,
constant for a given set): the returned schedule has the minimal f(b). Returns
`(Column, true period cost)`.
"""
function schedule_min_cost(fd::FluidData, x::Vector{Int}, S::Vector{Bool}, θ::Vector{Int};
                           k::Vector{Int} = fd.K0)
    any(S) || return empty_column(fd), lp_approx_cost(fd, x, empty_column(fd))
    BIG = 1e6
    FORCE = 1e3
    maxK = maximum(fd.K0)
    function schedule_of(set::Vector{Bool})
        λ = zeros(fd.M, 2, maxK + 1)
        for m in 1:fd.M
            λ[m, 2, :] .= set[m] ? fd.cc[m] - FORCE : BIG
            # instances with a free-day quota: forbid declaring more free days than the
            # remaining quota k of the pole (free days are otherwise used greedily, since they
            # only lower the period's penalty)
            # A tiny cost per declared free day breaks the ties in favour of not declaring
            # useless free days (otherwise the MILP may spend quota on a pole that has no
            # maintenance this period).
            for κ in qrange(fd.K0[fd.pole_of[m]])
                λ[m, :, κ + 1] .+= 1e-4 * κ
                κ > k[fd.pole_of[m]] && (λ[m, :, κ + 1] .= BIG)
            end
        end
        cand, _ = solve_pricing(fd, λ)
        (isnothing(cand) || any(set[m] && cand.start[m] == 0 for m in 1:fd.M)) && return nothing
        return cand
    end

    col = schedule_of(S)
    if isnothing(col)
        order = sort([m for m in 1:fd.M if S[m]]; by = m -> (-(x[m] - θ[m]), -fd.δ[m][x[m]]))
        kept = fill(false, fd.M)
        col = empty_column(fd)
        for m in order
            kept[m] = true
            c = schedule_of(kept)
            if isnothing(c)
                kept[m] = false
            else
                col = c
            end
        end
    end
    return col, lp_approx_cost(fd, x, col)
end

"""
Policy (x, k, t_global) -> (Column, solve_time) of baseline 1. Thresholds are optimised once
(at construction), schedules are computed on demand for each new joint state and cached in
memory and in `cache_file`. On instances with a free-day quota, the schedule may declare free
days, up to the remaining quota k of each pole, and the cache key also includes k.
"""
function make_policy_baseline1(fd::FluidData;
                               cache_file::Union{String,Nothing} = nothing,
                               log_solve::Function = info -> nothing,
                               gamma::Float64 = 1.0)
    θ, _ = optimize_thresholds(fd; gamma = gamma)
    @printf("  baseline1 thresholds θ_m (maintain iff state >= θ_m): %s\n", string(θ))

    cache, next_cache_id = load_cache(cache_file)

    function policy(x::Vector{Int}, k::Vector{Int}, t_global::Int)
        # instances with a quota: the schedule also depends on the remaining quota k
        key = fd.inst.simplified ? Tuple(x) : (Tuple(x), Tuple(k))
        if !haskey(cache, key)
            t_begin = time()
            S = [x[m] >= θ[m] for m in 1:fd.M]
            col, cost = schedule_min_cost(fd, x, S, θ; k = k)
            elapsed = time() - t_begin
            cache[key] = (col.start, col.free_days, elapsed, cost)
            t0_cyc = (t_global - 1) % N_PERIODS_PER_YEAR + 1
            log_solve((key = (Tuple(x), Tuple(k), t0_cyc, 1), time = elapsed, lp_value = cost,
                       iterations = 1, n_columns = 0, converged = true))
            append_cache_entry!(cache_file, next_cache_id, key, cache[key])
            return col, elapsed
        end
        return Column(cache[key][1], cache[key][2]), 0.0
    end
    return policy
end
