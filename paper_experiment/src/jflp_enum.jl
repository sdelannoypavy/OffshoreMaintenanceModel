# jflp_enum.jl
#
# JFLP (enum.) and JFLP-Q of paper.tex, solved with every joint action from the start (no column
# generation), and their fluid policies (Section "Fluid policies").
#
#   - JFLP (enum.): the Joint fluid LP (eq:fluid), the same LP as the master of jflp_cg.jl with every
#     column. The local state of a component contains the remaining quota of its substation.
#   - JFLP-Q: the Joint fluid LP with quota subsystem (eq:fluid-quota-subsystem, Appendix
#     "app:light"). The local state of a component is its degradation state, and every substation
#     has a quota subsystem (state k, action κ). The local cost is c̃_m(n, δ) = δ c^rep_m + T π ℓ_m(n).
#
# The joint actions are enumerated once per instance (joint_actions.jl). Only the joint action of least
# cost of every (set of maintained components, κ vector) is kept, since the other joint actions with
# the same local actions have the same coefficients in the LP and a larger cost.
# The policy applies the first-step joint action of largest weight μ*_{b,1}.

using JuMP, Gurobi, Printf

"Add the constraint Σ_i coefs_i vars_i == rhs through MOI, without building a JuMP expression (rows with many terms)."
function add_linear_eq!(model::Model, vars::Vector{VariableRef}, coefs::Vector{Float64}, rhs::Float64)
    terms = [MOI.ScalarAffineTerm(c, index(v)) for (v, c) in zip(vars, coefs)]
    MOI.add_constraint(backend(model), MOI.ScalarAffineFunction(terms, 0.0), MOI.EqualTo(rhs))
end

"""
Solve the LP. It is always feasible, but Gurobi's presolve occasionally declares it infeasible (observed on two_poles
with one thread): it is then solved again without presolve, then without presolve and with NumericFocus = 3.
"""
function optimize_with_fallback!(model::Model)
    optimize!(model)
    for params in (["Presolve" => 0], ["Presolve" => 0, "NumericFocus" => 3])
        is_solved_and_feasible(model) && return
        for (name, val) in params
            set_optimizer_attribute(model, name, val)
        end
        optimize!(model)
        is_solved_and_feasible(model) && println("  LP solved after a numerical failure, with ", params)
    end
    assert_is_solved_and_feasible(model)
end

"""
    solve_jflp_enum(fd, x0, k0, t0_cyc, H; lp_params, verbose)

Solves the Joint fluid LP (eq:fluid) for the initial state (x0, k0), the period of the year t0_cyc and
the fluid horizon H, with all the joint actions of the enumeration, and returns the first-step joint
action of largest weight μ*_{b,1}, with statistics. Same LP as the master of jflp_cg.jl with every column.
"""
function solve_jflp_enum(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, t0_cyc::Int, H::Int;
                         lp_params::Vector{<:Pair} = Pair{String,Int}[], verbose::Bool = false)
    t_begin = time()
    E = joint_action_enum(fd)
    t_first = time()
    c_set, c_kappa, c_sched, c_cost = first_step_enum(fd, E, x0, k0)
    t_first = time() - t_first
    M = fd.M; K = fd.inst.n_poles
    pole_of = fd.pole_of
    mode_of(set, m) = E.sets[set][m] ? 2 : 1

    t_build = time()
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    for (name, val) in lp_params
        set_optimizer_attribute(model, name, val)
    end

    # variables q^m_{(n,k),(mode,κ),t}, t ≥ 2, κ ≤ k (local state-action constraint)
    q = Dict{NTuple{6,Int},VariableRef}()
    obj = AffExpr(0.0)
    for m in 1:M, t in 2:H, n in 1:fd.n_states[m], mode in 1:2
        for k in qrange(fd.K0[pole_of[m]]), κ in qrange(k)
            v = @variable(model, lower_bound = 0)
            q[(m, n, mode, k, κ, t)] = v
            add_to_expression!(obj, local_cost(fd, m, n, mode, κ), v)
        end
    end
    μ1 = @variable(model, [1:length(c_cost)], lower_bound = 0)
    nb = length(E.col_cost)
    μ = [@variable(model, [1:nb], lower_bound = 0) for t in 2:H]
    @objective(model, Min, obj)
    set_objective_coefficient(model, μ1, c_cost)
    for t in 2:H
        set_objective_coefficient(model, μ[t - 1], E.col_cost)
    end

    # Σ_b μ_{b,1} = 1 (eq:fluid-first)
    add_linear_eq!(model, μ1, ones(length(μ1)), 1.0)

    # initial condition (eq:fluid-initial): Σ_a q^m_{(n,k),a,2} = Σ_b p_m((n,k) | x0_m, A_m(b)) μ_{b,1}
    if H >= 2
        rows = Dict{NTuple{3,Int},Tuple{Vector{VariableRef},Vector{Float64}}}()
        for m in 1:M, n in 1:fd.n_states[m], k in qrange(fd.K0[pole_of[m]])
            rows[(m, n, k)] = ([q[(m, n, mode, k, κ, 2)] for mode in 1:2 for κ in qrange(k)],
                               ones(2 * length(qrange(k))))
        end
        for b in eachindex(c_cost)
            κ = E.kappas[c_kappa[b]]
            for m in 1:M
                p = pole_of[m]; mode = mode_of(c_set[b], m)
                k_next = quota_next(1, t0_cyc, k0[p], κ[p], fd.K0[p])
                for n in 1:fd.n_states[m]
                    prob = trans_prob(fd, m, n, x0[m], mode)
                    prob == 0.0 && continue
                    push!(rows[(m, n, k_next)][1], μ1[b]); push!(rows[(m, n, k_next)][2], -prob)
                end
            end
        end
        for (vars, coefs) in values(rows)
            add_linear_eq!(model, vars, coefs, 0.0)
        end
    end

    # local flow constraints (eq:fluid-flow), t ≥ 3
    for t in 3:H, m in 1:M
        K0 = fd.K0[pole_of[m]]
        for n in 1:fd.n_states[m], kn in qrange(K0)
            lhs = sum(q[(m, n, mode, kn, κ, t)] for mode in 1:2, κ in qrange(kn))
            preds = quota_predecessors(t, t0_cyc, kn, K0)
            rhs = AffExpr(0.0)
            for np in 1:fd.n_states[m], mode in 1:2
                prob = trans_prob(fd, m, n, np, mode)
                prob == 0.0 && continue
                for (kp, κp) in preds
                    add_to_expression!(rhs, prob, q[(m, np, mode, kp, κp, t - 1)])
                end
            end
            @constraint(model, lhs == rhs)
        end
    end

    # consistency (eq:fluid-consistency), t ≥ 2: Σ_x q^m_{x,a,t} = Σ_{b | A_m(b) = a} μ_{b,t}
    if H >= 2
        cols_of = Dict{NTuple{3,Int},Vector{Int}}()   # (m, mode, κ) -> columns b with A_m(b) = (mode, κ)
        for m in 1:M, mode in 1:2, κ in qrange(fd.K0[pole_of[m]])
            cols_of[(m, mode, κ)] = Int[]
        end
        for b in 1:nb
            κ = E.kappas[E.col_kappa[b]]
            for m in 1:M
                push!(cols_of[(m, mode_of(E.col_set[b], m), κ[pole_of[m]])], b)
            end
        end
        for t in 2:H, ((m, mode, κ), bs) in cols_of
            K0 = fd.K0[pole_of[m]]
            qs = [q[(m, n, mode, k, κ, t)] for n in 1:fd.n_states[m] for k in κ:STEP:K0]
            add_linear_eq!(model, vcat(qs, μ[t - 1][bs]), vcat(ones(length(qs)), fill(-1.0, length(bs))), 0.0)
        end
    end

    # quota consistency (eq:fluid-copies), t ≥ 2
    for p in 1:K, m in fd.comps_of_pole[p][2:end], t in 2:H
        m1 = fd.comps_of_pole[p][1]; K0 = fd.K0[p]
        for k in qrange(K0), κ in qrange(k)
            @constraint(model,
                sum(q[(m, n, mode, k, κ, t)] for n in 1:fd.n_states[m], mode in 1:2) ==
                sum(q[(m1, n, mode, k, κ, t)] for n in 1:fd.n_states[m1], mode in 1:2))
        end
    end
    t_build = time() - t_build

    t_solve = time()
    optimize_with_fallback!(model)
    t_solve = time() - t_solve

    b1 = argmax(value.(μ1))
    best = first_step_column(fd, E, x0, c_sched[b1], c_kappa[b1])
    info = (time = time() - t_begin, lp_value = objective_value(model),
            t_enum = E.time, t_first = t_first, t_build = t_build, t_solve = t_solve,
            n_columns = length(μ1) + (H - 1) * nb, n_first = length(μ1), n_per_stage = nb,
            n_scheds = length(E.scheds), n_sets = length(E.sets),
            n_rows = num_constraints(model; count_variable_in_set_constraints = false),
            iterations = 0, converged = true)
    verbose && @printf("  enum %.2fs (once)  first step %.2fs  build %.2fs  solve %.2fs  LP %.6f  columns %d\n",
                       E.time, t_first, t_build, t_solve, info.lp_value, info.n_columns)
    return best, info
end

"""
    solve_jflp_q_enum(fd, x0, k0, t0_cyc, H; verbose)

Solves the Joint fluid LP with quota subsystem (eq:fluid-quota-subsystem) with all the joint actions
of the enumeration. Variables q^m_{n,δ,t} of the components, without the quota, with the cost
c̃_m(n, δ) = δ c^rep_m + T π ℓ_m(n), r^s_{k,κ,t} (κ ≤ k) of the quota subsystem of substation s,
without cost, and μ_{b,t}, μ_{b,1} as in solve_jflp_enum. Returns the first-step joint action of
largest weight μ*_{b,1}, with statistics.
"""
function solve_jflp_q_enum(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, t0_cyc::Int, H::Int;
                           verbose::Bool = false)
    t_begin = time()
    E = joint_action_enum(fd)
    t_first = time()
    c_set, c_kappa, c_sched, c_cost = first_step_enum(fd, E, x0, k0)
    t_first = time() - t_first
    M = fd.M; K = fd.inst.n_poles; pole_of = fd.pole_of
    T = fd.inst.T_period; π = constant_daily_penalty(fd)
    mode_of(set, m) = E.sets[set][m] ? 2 : 1
    nb = length(E.col_cost)

    t_build = time()
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    # q^m_{n,δ,t}, t ≥ 2, with c̃_m(n, δ) = δ Rc_m + T π ℓ_m(n) (eq:fluid-quota-subsystem-obj)
    q = Dict{NTuple{4,Int},VariableRef}()
    obj = AffExpr(0.0)
    for m in 1:M, t in 2:H, n in 1:fd.n_states[m], mode in 1:2
        v = @variable(model, lower_bound = 0)
        q[(m, n, mode, t)] = v
        add_to_expression!(obj, (mode == 2 ? fd.cc[m] : 0.0) + T * π * capacity_loss(fd, m, n), v)
    end
    # r^s_{k,κ,t}, t ≥ 2, κ ≤ k (eq:fluid-quota-subsystem-qconsistency), no cost
    r = Dict{NTuple{4,Int},VariableRef}()
    for s in 1:K, t in 2:H, k in qrange(fd.K0[s]), κ in qrange(k)
        r[(s, k, κ, t)] = @variable(model, lower_bound = 0)
    end
    μ1 = @variable(model, [1:length(c_cost)], lower_bound = 0)
    μ = [@variable(model, [1:nb], lower_bound = 0) for t in 2:H]
    @objective(model, Min, obj)
    set_objective_coefficient(model, μ1, c_cost)
    for t in 2:H
        set_objective_coefficient(model, μ[t - 1], E.col_cost)
    end

    add_linear_eq!(model, μ1, ones(length(μ1)), 1.0)   # Σ_b μ_{b,1} = 1
    if H >= 2
        # initial conditions of the components (eq:fluid-quota-subsystem-initial) and of the quotas
        # (eq:fluid-quota-subsystem-qinitial)
        rows = Dict{Tuple{Symbol,Int,Int},Tuple{Vector{VariableRef},Vector{Float64}}}()
        for m in 1:M, n in 1:fd.n_states[m]
            rows[(:comp, m, n)] = ([q[(m, n, mode, 2)] for mode in 1:2], [1.0, 1.0])
        end
        for s in 1:K, k in qrange(fd.K0[s])
            rows[(:quota, s, k)] = ([r[(s, k, κ, 2)] for κ in qrange(k)], ones(length(qrange(k))))
        end
        for b in eachindex(c_cost)
            κ = E.kappas[c_kappa[b]]
            for m in 1:M, n in 1:fd.n_states[m]
                prob = trans_prob(fd, m, n, x0[m], mode_of(c_set[b], m))
                prob == 0.0 && continue
                push!(rows[(:comp, m, n)][1], μ1[b]); push!(rows[(:comp, m, n)][2], -prob)
            end
            for s in 1:K
                key = (:quota, s, quota_next(1, t0_cyc, k0[s], κ[s], fd.K0[s]))
                push!(rows[key][1], μ1[b]); push!(rows[key][2], -1.0)
            end
        end
        for (vars, coefs) in values(rows)
            add_linear_eq!(model, vars, coefs, 0.0)
        end
    end
    # flows of the components (eq:fluid-quota-subsystem-flow) and of the quotas (eq:fluid-quota-subsystem-qflow), t ≥ 3
    for t in 3:H
        for m in 1:M, n in 1:fd.n_states[m]
            rhs = AffExpr(0.0)
            for np in 1:fd.n_states[m], mode in 1:2
                prob = trans_prob(fd, m, n, np, mode)
                prob == 0.0 || add_to_expression!(rhs, prob, q[(m, np, mode, t - 1)])
            end
            @constraint(model, q[(m, n, 1, t)] + q[(m, n, 2, t)] == rhs)
        end
        for s in 1:K, kn in qrange(fd.K0[s])
            preds = quota_predecessors(t, t0_cyc, kn, fd.K0[s])
            @constraint(model, sum(r[(s, kn, κ, t)] for κ in qrange(kn)) ==
                               sum((r[(s, kp, κp, t - 1)] for (kp, κp) in preds); init = AffExpr(0.0)))
        end
    end
    # consistency of the components (eq:fluid-quota-subsystem-consistency) and of the quotas
    # (eq:fluid-quota-subsystem-qconsistency), t ≥ 2
    if H >= 2
        cols_of = Dict{Tuple{Symbol,Int,Int},Vector{Int}}()
        for b in 1:nb
            κ = E.kappas[E.col_kappa[b]]
            for m in 1:M
                push!(get!(cols_of, (:comp, m, mode_of(E.col_set[b], m)), Int[]), b)
            end
            for s in 1:K
                push!(get!(cols_of, (:quota, s, κ[s]), Int[]), b)
            end
        end
        for t in 2:H
            for m in 1:M, mode in 1:2
                bs = get(cols_of, (:comp, m, mode), Int[])
                qs = [q[(m, n, mode, t)] for n in 1:fd.n_states[m]]
                add_linear_eq!(model, vcat(qs, μ[t - 1][bs]), vcat(ones(length(qs)), fill(-1.0, length(bs))), 0.0)
            end
            for s in 1:K, κ in qrange(fd.K0[s])
                bs = get(cols_of, (:quota, s, κ), Int[])
                rs = [r[(s, k, κ, t)] for k in κ:STEP:fd.K0[s]]
                add_linear_eq!(model, vcat(rs, μ[t - 1][bs]), vcat(ones(length(rs)), fill(-1.0, length(bs))), 0.0)
            end
        end
    end
    t_build = time() - t_build

    t_solve = time()
    optimize_with_fallback!(model)
    t_solve = time() - t_solve

    b1 = argmax(value.(μ1))
    best = first_step_column(fd, E, x0, c_sched[b1], c_kappa[b1])
    info = (time = time() - t_begin, lp_value = objective_value(model),
            t_enum = E.time, t_first = t_first, t_build = t_build, t_solve = t_solve,
            n_columns = length(μ1) + (H - 1) * nb, iterations = 0, converged = true)
    verbose && @printf("  quota subsystem: first step %.2fs  build %.2fs  solve %.2fs  LP %.6f  columns %d\n",
                       t_first, t_build, t_solve, info.lp_value, info.n_columns)
    return best, info
end

# ─────────────────────────────────────────────────────────────────────────────
# Fluid policies
# ─────────────────────────────────────────────────────────────────────────────

"""
Fluid policy of JFLP (enum.), or of JFLP-Q if `quota_subsystem`, with fluid horizon `H`
(Section "Fluid policies"). See make_fluid_policy (model.jl) for `cache_file` and `log_solve`.
"""
function make_policy_jflp_enum(fd::FluidData; H::Int = 6, quota_subsystem::Bool = false,
                               cache_file::Union{String,Nothing} = nothing, log_solve::Function = info -> nothing)
    function solve(x, k, t0_cyc, H_eff)
        col, info = quota_subsystem ? solve_jflp_q_enum(fd, x, k, t0_cyc, H_eff) : solve_jflp_enum(fd, x, k, t0_cyc, H_eff)
        return col, merge(info, (t_milp = 0.0, t_lp = info.t_solve))
    end
    return make_fluid_policy(fd, solve; H = H, cache_file = cache_file, log_solve = log_solve)
end
