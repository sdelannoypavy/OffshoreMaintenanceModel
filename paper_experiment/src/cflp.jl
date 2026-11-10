# cflp.jl
#
# CFLP of paper.tex: the Compact fluid LP (eq:compact-fluid) and its fluid policy (Section
# "Fluid policies").
#
#   - Variables q^m_{x,a,t}, t ∈ [1:H], with x = (n, k), a = (mode, κ) and κ ≤ k, with the initial
#     condition (eq:compact-fluid-initial), the flow, the consistency of the quota between the
#     components of a substation, and the separable cost c̃_m (local_cost, model.jl) at t ≥ 2.
#   - For every t, the vector b̄_t of the means of the variables of the MILP formulation of B
#     (Appendix "MILP formulation of the joint action cost minimization"), with {0,1} relaxed into
#     [0,1]: b̄_t ∈ B̄ for t ≥ 2, with the joint action cost f(b̄_t), and b̄_1 ∈ B̄(x0), with the quota
#     constraint κ ≤ k0 and the true cost c⊗(x0, b̄_1).
#   - Consistency (eq:compact-delta) and (eq:compact-chi), for every t: the probability that m is
#     maintained is δ_m(b̄_t), and the probability that κ days are declared on the substation of m is
#     χ^{s(m)}_κ(b̄_t).
#   - Policy: the components with δ_m(b̄*_1) > 1/2 are maintained, scheduled by `round_schedule`.

using JuMP, Gurobi, Printf

"""
    solve_cflp(fd, x0, k0, t0_cyc, H)

Builds and solves the Compact fluid LP for the initial state (x0, k0), the period of the year t0_cyc
and the fluid horizon H. Returns the joint action of the policy, the optimal value and the time of
the LP solve alone.
"""
function solve_cflp(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, t0_cyc::Int, H::Int)
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
            t >= 2 && add_to_expression!(obj, local_cost(fd, m, n, mode, κ), v)
        end
    end
    # initial condition (eq:compact-fluid-initial)
    for m in 1:M, n in 1:fd.n_states[m], k in qrange(fd.K0[fd.pole_of[m]])
        @constraint(model, sum(q[(m, n, mode, k, κ, 1)] for mode in 1:2, κ in qrange(k)) ==
                           (n == x0[m] && k == k0[fd.pole_of[m]] ? 1.0 : 0.0))
    end
    # flow, t ≥ 2
    for t in 2:H, m in 1:M
        K0 = fd.K0[fd.pole_of[m]]
        for n in 1:fd.n_states[m], kn in qrange(K0)
            rhs = AffExpr(0.0)
            for np in 1:fd.n_states[m], mode in 1:2
                prob = trans_prob(fd, m, n, np, mode)
                prob < 1e-12 && continue
                for (kp, κp) in quota_predecessors(t, t0_cyc, kn, K0)
                    add_to_expression!(rhs, prob, q[(m, np, mode, kp, κp, t - 1)])
                end
            end
            @constraint(model, sum(q[(m, n, mode, kn, κ, t)] for mode in 1:2, κ in qrange(kn)) == rhs)
        end
    end
    # consistency of the quota between the components of a substation, every t
    for p in 1:K, m in fd.comps_of_pole[p][2:end], t in 1:H
        m1 = fd.comps_of_pole[p][1]; K0 = fd.K0[p]
        for k in qrange(K0), κ in qrange(k)
            @constraint(model,
                sum(q[(m, n, mode, k, κ, t)] for n in 1:fd.n_states[m], mode in 1:2) ==
                sum(q[(m1, n, mode, k, κ, t)] for n in 1:fd.n_states[m1], mode in 1:2))
        end
    end

    # relaxed MILP formulation of B (t ≥ 2) and of B(x0) (t = 1), one copy per t
    starts = [start_days(fd, m) for m in 1:M]
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
        for τ in FIRST_START_DAY:STEP:T
            @constraint(model, sum(fd.rho[m] * ong(m, τ) for m in 1:M) <= inst.R)   # technician capacity
        end
        any_m = @variable(model, [1:K, 1:T], lower_bound = 0, upper_bound = 1)
        free = @variable(model, [1:K, 1:T], lower_bound = 0, upper_bound = 1)
        for p in 1:K
            ksel = [@variable(model, lower_bound = 0, upper_bound = 1) for κ in qrange(kcap[p])]   # χ^p_κ
            @constraint(model, sum(ksel) == 1)
            @constraint(model, sum(free[p, τ] for τ in 1:T) == sum(κ * ksel[i] for (i, κ) in enumerate(qrange(kcap[p]))))
            # consistency (eq:compact-chi): P(κ^{s(m)} = κ) of every component m of p = χ^p_κ(b̄_t)
            for m in fd.comps_of_pole[p], (i, κ) in enumerate(qrange(kcap[p]))
                @constraint(model, sum(q[(m, n, mode, k, κ, t)] for n in 1:fd.n_states[m], mode in 1:2,
                                       k in κ:STEP:fd.K0[p]) == ksel[i])
            end
            for m in fd.comps_of_pole[p], τ in 1:T
                @constraint(model, any_m[p, τ] >= ong(m, τ))
            end
        end
        # consistency (eq:compact-delta): P(m maintained) = δ_m(b̄_t)
        for m in 1:M
            K0 = fd.K0[fd.pole_of[m]]
            @constraint(model, sum(q[(m, n, 2, k, κ, t)] for n in 1:fd.n_states[m], k in qrange(K0), κ in qrange(k)) ==
                               z[(m, t)])
        end
        if t >= 2 || inst.simplified
            # joint action cost f(b̄_t), w = any_m ∧ free
            w = @variable(model, [1:K, 1:T], lower_bound = 0, upper_bound = 1)
            for p in 1:K, τ in 1:T
                @constraint(model, w[p, τ] <= free[p, τ]); @constraint(model, w[p, τ] <= any_m[p, τ])
                pen = inst.poles[p].daily_penalty[τ]
                add_to_expression!(obj, pen, any_m[p, τ]); add_to_expression!(obj, -pen, w[p, τ])
            end
        else
            # true cost of the full model, linearised as in the first-step pricing problem (jflp_cg.jl)
            L = @variable(model, [1:K, 1:T], lower_bound = 0)
            u = @variable(model, [1:K, 1:T], lower_bound = 0)
            v = @variable(model, [1:K, 1:T], lower_bound = 0)
            for p in 1:K, τ in 1:T
                for m in fd.comps_of_pole[p]
                    ℓ0 = capacity_loss(fd, m, x0[m]); ℓ1 = capacity_loss(fd, m, 1)
                    @constraint(model, L[p, τ] >= ℓ0 - (ℓ0 - ℓ1) * done(m, τ))
                end
                @constraint(model, u[p, τ] >= any_m[p, τ] - free[p, τ])
                @constraint(model, v[p, τ] >= L[p, τ] - any_m[p, τ] - free[p, τ])
                pen = inst.poles[p].daily_penalty[τ]
                add_to_expression!(obj, pen, u[p, τ]); add_to_expression!(obj, pen, v[p, τ])
            end
        end
        if t == 1
            # replacement costs of the first step, and failure costs of the simplified model (constant)
            for m in 1:M
                add_to_expression!(obj, fd.cc[m], z[(m, 1)])
            end
            inst.simplified && add_to_expression!(obj, sum(failure_cost(fd, m, x0[m]) for m in 1:M))
        end
    end
    @objective(model, Min, obj)
    t_solve = time()
    optimize!(model)
    if !is_solved_and_feasible(model)
        # Gurobi's presolve occasionally declares this feasible LP infeasible (observed on one_pole):
        # solve it again without presolve
        set_optimizer_attribute(model, "Presolve", 0)
        optimize!(model)
        is_solved_and_feasible(model) && println("  compact LP solved without presolve after a numerical failure")
    end
    assert_is_solved_and_feasible(model)
    t_solve = time() - t_solve   # time of the LP solve alone (the rest is the construction of the LP)

    δ1 = [value(z[(m, 1)]) for m in 1:M]
    return round_schedule(fd, x0, k0, δ1 .> 0.5), objective_value(model), t_solve
end

"""
    round_schedule(fd, x0, k0, maintain)

Admissible joint action in the state (x0, k0) that maintains the components of `maintain` that fit
(Section "Fluid policies"). The components are considered from the most degraded, and every one
starts on the earliest day of the start-day grid compatible with the technician capacity, a
component that does not fit being left unmaintained. Every substation then declares as many days as
its remaining quota allows, without exceeding its number of days with a positive capacity loss, and
places them on its days of largest loss, that is, on its maintenance days first.
"""
function round_schedule(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, maintain::AbstractVector{Bool})
    inst = fd.inst; M = fd.M; T = inst.T_period; K = inst.n_poles
    cand = [m for m in 1:M if maintain[m] && !isempty(start_days(fd, m))]
    sort!(cand; by = m -> (-capacity_loss(fd, m, x0[m]), -fd.rho[m]))
    start = zeros(Int, M); demand = zeros(T)
    for m in cand
        for τ in start_days(fd, m)
            if all(demand[s] + fd.rho[m] <= inst.R + 1e-9 for s in τ:(τ + fd.d[m] - 1))
                start[m] = τ
                for s in τ:(τ + fd.d[m] - 1); demand[s] += fd.rho[m]; end
                break
            end
        end
    end
    free_days = Vector{Vector{Int}}(undef, K)
    for p in 1:K
        ms = fd.comps_of_pole[p]
        loss = [any(under_repair(fd, start, m, τ) for m in ms) ? 1.0 :
                maximum(capacity_loss(fd, m, (start[m] > 0 && τ >= start[m] + fd.d[m]) ? 1 : x0[m]) for m in ms)
                for τ in 1:T]
        κ = STEP * div(min(k0[p], count(>(1e-12), loss)), STEP)
        free_days[p] = sort(sortperm(loss; rev = true)[1:κ])
    end
    return Column(start, free_days)
end

"Fluid policy of CFLP with fluid horizon `H` (Section \"Fluid policies\"). See make_fluid_policy (model.jl)."
function make_policy_cflp(fd::FluidData; H::Int = 6,
                          cache_file::Union{String,Nothing} = nothing, log_solve::Function = info -> nothing)
    function solve(x, k, t0_cyc, H_eff)
        t_begin = time()
        col, lp_value, t_solve = solve_cflp(fd, x, k, t0_cyc, H_eff)
        return col, (time = time() - t_begin, lp_value = lp_value, t_milp = 0.0, t_lp = t_solve)
    end
    return make_fluid_policy(fd, solve; H = H, cache_file = cache_file, log_solve = log_solve)
end
