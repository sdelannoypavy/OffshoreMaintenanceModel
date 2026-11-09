# compact_paper.jl
#
# Compact fluid LP of Section 3.3 of paper.tex (eq:compact-fluid), and its policy (Section 3.4).
# This file follows the paper exactly when QUOTA_STEP = 1 (default, instances.jl). With
# QUOTA_STEP = 15, the quota and the start days are discretised in steps of 15 days.
#
#   - Variables q^m_{x,a,t}, t ∈ [1:H], with x = (n, k), a = (mode, κ) and κ ≤ k (local
#     state-action constraint), with the initial condition (eq:compact-fluid-initial), the local
#     flow, the quota consistency, and the separable approximation c̃_m of the cost at t ≥ 2.
#   - For every t, the vector b̄_t of the means of the variables of the MILP formulation of B
#     (Appendix C), with {0,1} relaxed into [0,1]: b̄_t ∈ B̄ for t ≥ 2, with the joint action cost
#     f(b̄_t), and b̄_1 ∈ B̄(x0), which adds the quota constraint κ ≤ k0 and the auxiliary variables
#     of the true cost c⊗(x0, ·) (Section 2.2), with the cost c⊗(x0, b̄_1).
#   - Consistency (eq:compact-fluid-consistency)-(eq:compact-fluid-consistency-kappa), for every
#     t: the probability that m is maintained is δ_m(b̄_t), and the expected number of penalty-free
#     days of the pole of m is κ^{s(m)}(b̄_t).
#   - Policy: the components with δ_m(b̄_1*) > 1/2 are maintained, scheduled by `schedule_paper`
#     (fluid_cg_paper.jl).
#
# Requires fluid_lp_cg.jl and fluid_cg_paper.jl to be loaded first.

using JuMP, Gurobi, Printf

"""
    solve_compact_paper(fd, x0, k0, t0_cyc, H; kappa_dist = false)

Builds and solves the Compact fluid LP for the initial state (x0, k0), and returns the joint
action of the policy, the optimal value, the relaxed maintenance decisions δ_m(b̄_1*) and the time
of the LP solve alone. With `kappa_dist = true`, the consistency of the number of penalty-free days
(eq:compact-fluid-consistency-kappa) is written on its distribution instead of its mean: the number
of days is encoded by the indicators κ^sel_{p,κ} (Appendix C), whose means in b̄_t are the
probabilities of the values κ.
"""
function solve_compact_paper(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, t0_cyc::Int, H::Int;
                             kappa_dist::Bool = false)
    inst = fd.inst; M = fd.M; K = inst.n_poles; T = inst.T_period
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(model)

    q = Dict{NTuple{6,Int},VariableRef}()
    obj = AffExpr(0.0)
    for m in 1:M, t in 1:H, n in 1:fd.n_states[m], mode in 1:2
        K0 = fd.K0[fd.pole_of[m]]
        for k in qrange(K0), κ in qrange(k)   # κ ≤ k: local state-action constraint
            v = @variable(model, lower_bound = 0)
            q[(m, n, mode, k, κ, t)] = v
            t >= 2 && add_to_expression!(obj, paper_local_cost(fd, m, n, mode, κ), v)
        end
    end
    # initial condition (eq:compact-fluid-initial)
    for m in 1:M, n in 1:fd.n_states[m], k in qrange(fd.K0[fd.pole_of[m]])
        p = fd.pole_of[m]
        @constraint(model, sum(q[(m, n, mode, k, κ, 1)] for mode in 1:2, κ in qrange(k)) ==
                           (n == x0[m] && k == k0[p] ? 1.0 : 0.0))
    end
    # local flow, t ≥ 2
    for t in 2:H, m in 1:M
        K0 = fd.K0[fd.pole_of[m]]
        for n in 1:fd.n_states[m], kn in qrange(K0)
            preds = quota_predecessors(t, t0_cyc, kn, K0)
            rhs = AffExpr(0.0)
            for np in 1:fd.n_states[m], mode in 1:2
                prob = trans_prob(fd, m, n, np, mode)
                prob < 1e-12 && continue
                for (kp, κp) in preds
                    add_to_expression!(rhs, prob, q[(m, np, mode, kp, κp, t - 1)])
                end
            end
            @constraint(model, sum(q[(m, n, mode, kn, κ, t)] for mode in 1:2, κ in qrange(kn)) == rhs)
        end
    end
    # quota consistency, every t
    for p in 1:K, m in fd.comps_of_pole[p][2:end], t in 1:H
        m1 = fd.comps_of_pole[p][1]; K0 = fd.K0[p]
        for k in qrange(K0), κ in qrange(k)
            @constraint(model,
                sum(q[(m, n, mode, k, κ, t)] for n in 1:fd.n_states[m], mode in 1:2) ==
                sum(q[(m1, n, mode, k, κ, t)] for n in 1:fd.n_states[m1], mode in 1:2))
        end
    end

    # relaxed MILP formulation of B (t ≥ 2) and of B(x0) (t = 1), one copy per t
    last_start = [T - fd.d[m] + 1 - inst.margin for m in 1:M]
    first_start = FIRST_START_DAY
    starts = [last_start[m] >= first_start ? collect(first_start:QUOTA_STEP:last_start[m]) : Int[] for m in 1:M]
    z = Dict{Tuple{Int,Int},VariableRef}()
    for t in 1:H
        kcap = t == 1 ? k0 : fd.K0
        ms = Dict{Int,Dict{Int,VariableRef}}()
        for m in 1:M
            z[(m, t)] = @variable(model, lower_bound = 0, upper_bound = 1)
            ms[m] = Dict(τ => @variable(model, lower_bound = 0, upper_bound = 1) for τ in starts[m])
            @constraint(model, sum(values(ms[m]); init = AffExpr(0.0)) == z[(m, t)])
        end
        ong(m, τ) = sum((ms[m][s] for s in starts[m] if s <= τ <= s + fd.d[m] - 1); init = AffExpr(0.0))
        done(m, τ) = sum((ms[m][s] for s in starts[m] if τ >= s + fd.d[m]); init = AffExpr(0.0))
        for τ in first_start:QUOTA_STEP:T
            @constraint(model, sum(fd.rho[m] * ong(m, τ) for m in 1:M) <= inst.R)
        end
        any_m = @variable(model, [1:K, 1:T], lower_bound = 0, upper_bound = 1)
        free = @variable(model, [1:K, 1:T], lower_bound = 0, upper_bound = 1)
        for p in 1:K
            ksel = [@variable(model, lower_bound = 0, upper_bound = 1) for κ in qrange(kcap[p])]
            @constraint(model, sum(ksel) == 1)
            @constraint(model, sum(free[p, τ] for τ in 1:T) == sum(κ * ksel[i] for (i, κ) in enumerate(qrange(kcap[p]))))
            for m in fd.comps_of_pole[p]
                K0 = fd.K0[p]
                if kappa_dist
                    # the mean of the indicator κ^sel_{p,κ} is the probability that κ days are declared:
                    # P(κ^{s(m)} = κ) of each component = mean of κ^sel_{p,κ}
                    for (i, κ) in enumerate(qrange(kcap[p]))
                        @constraint(model, sum(q[(m, n, mode, k, κ, t)] for n in 1:fd.n_states[m], mode in 1:2,
                                               k in κ:QUOTA_STEP:K0) == ksel[i])
                    end
                else
                    # consistency (eq:compact-fluid-consistency-kappa): E[κ] of each component = κ^{s(m)}(b̄_t)
                    @constraint(model, sum(κ * q[(m, n, mode, k, κ, t)] for n in 1:fd.n_states[m], mode in 1:2,
                                           k in qrange(K0), κ in qrange(k)) == sum(free[p, τ] for τ in 1:T))
                end
            end
        end
        # consistency (eq:compact-fluid-consistency): P(m maintained) = δ_m(b̄_t)
        for m in 1:M
            K0 = fd.K0[fd.pole_of[m]]
            @constraint(model, sum(q[(m, n, 2, k, κ, t)] for n in 1:fd.n_states[m], k in qrange(K0), κ in qrange(k)) ==
                               z[(m, t)])
        end
        if t == 1
            # true cost c⊗(x0, b̄_1), linearised as in the first-step pricing problem
            L = @variable(model, [1:K, 1:T], lower_bound = 0)
            u = @variable(model, [1:K, 1:T], lower_bound = 0)
            v = @variable(model, [1:K, 1:T], lower_bound = 0)
            for p in 1:K, τ in 1:T
                for m in fd.comps_of_pole[p]
                    @constraint(model, any_m[p, τ] >= ong(m, τ))
                    ℓ0 = capacity_loss(fd, m, x0[m]); ℓ1 = capacity_loss(fd, m, 1)
                    @constraint(model, L[p, τ] >= ℓ0 - (ℓ0 - ℓ1) * done(m, τ))
                end
                @constraint(model, u[p, τ] >= any_m[p, τ] - free[p, τ])
                @constraint(model, v[p, τ] >= L[p, τ] - any_m[p, τ] - free[p, τ])
                pen = inst.poles[p].daily_penalty[τ]
                add_to_expression!(obj, pen, u[p, τ]); add_to_expression!(obj, pen, v[p, τ])
            end
            for m in 1:M
                add_to_expression!(obj, fd.cc[m], z[(m, 1)])
            end
        else
            # joint action cost f(b̄_t) = Mc
            w = @variable(model, [1:K, 1:T], lower_bound = 0, upper_bound = 1)
            for p in 1:K, τ in 1:T
                for m in fd.comps_of_pole[p]
                    @constraint(model, any_m[p, τ] >= ong(m, τ))
                end
                @constraint(model, w[p, τ] <= free[p, τ]); @constraint(model, w[p, τ] <= any_m[p, τ])
                pen = inst.poles[p].daily_penalty[τ]
                add_to_expression!(obj, pen, any_m[p, τ]); add_to_expression!(obj, -pen, w[p, τ])
            end
        end
    end
    @objective(model, Min, obj)
    t_solve = time()
    optimize!(model)
    if !is_solved_and_feasible(model)
        # Gurobi's presolve occasionally declares this feasible LP infeasible (observed on one_pole,
        # x = 1-1-1-2-2-1-1, k = 15, t0 = 6): solve it again without presolve
        set_optimizer_attribute(model, "Presolve", 0)
        optimize!(model)
        is_solved_and_feasible(model) && println("  compact LP solved without presolve after a numerical failure")
    end
    assert_is_solved_and_feasible(model)
    t_solve = time() - t_solve   # time of the LP solve alone (the rest is the construction of the LP)

    δ1 = [value(z[(m, 1)]) for m in 1:M]
    col = schedule_paper(fd, x0, k0, δ1 .> 0.5)
    return col, objective_value(model), δ1, t_solve
end

"Policy of the Compact fluid LP (Section 3.4): (x, k, t_global) -> (Column, solve_time), with a cache."
function make_policy_compact_paper(fd::FluidData; H::Int = 6, kappa_dist::Bool = false,
                                   cache_file::Union{String,Nothing} = nothing,
                                   log_solve::Function = info -> nothing)
    T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
    cache, next_cache_id = load_cache(cache_file)
    function policy(x::Vector{Int}, k::Vector{Int}, t_global::Int)
        t0_cyc = (t_global - 1) % N_PERIODS_PER_YEAR + 1
        H_eff = min(H, T_total - t_global + 1)
        key = (Tuple(x), Tuple(k), t0_cyc, H_eff)
        if !haskey(cache, key)
            t_begin = time()
            col, lp_value, _, t_solve = solve_compact_paper(fd, x, k, t0_cyc, H_eff; kappa_dist = kappa_dist)
            elapsed = time() - t_begin
            cache[key] = (col.start, col.free_days, elapsed, lp_value)
            log_solve((key = key, time = elapsed, lp_value = lp_value, iterations = 1, n_columns = 0,
                       converged = true, t_milp = 0.0, t_lp = t_solve))
            append_cache_entry!(cache_file, next_cache_id, key, cache[key])
            return col, elapsed
        end
        return Column(cache[key][1], cache[key][2]), 0.0
    end
    return policy
end
