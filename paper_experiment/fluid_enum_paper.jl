# fluid_enum_paper.jl
#
# Full fluid LP of Section 3.2 of paper.tex (the order-1 LP of fluid_cg_paper.jl, same variables q,
# same constraints, same costs), solved WITHOUT column generation: every joint action is enumerated
# once and the LP is solved with all its variables μ_{b,t} from the start.
#
#   - Joint actions: the sets S of components that can be maintained in the same period (start days
#     on the grid FIRST_START_DAY, FIRST_START_DAY + QUOTA_STEP, ... as in the pricing problems of
#     fluid_cg_paper.jl, technician capacity checked on every day), times the vectors (κ^s)_s of
#     numbers of declared penalty-free days, κ^s ∈ qrange(K0). For each (S, κ), only the schedule and
#     the free days of least cost are kept: the other joint actions with the same local actions
#     A_m(b) have the same coefficients in the LP and a larger cost, so their μ is 0 at the optimum.
#   - Both costs include the common mobilization cost MOBILIZATION_COST (instances.jl) of the
#     joint actions that maintain at least one component.
#   - t ≥ 2: cost f(b) = Mc (column_penalty, fluid_lp_cg.jl). For a fixed schedule, the best free
#     days of pole s are its κ^s days of largest π_τ m^s_τ. These columns do not depend on the
#     state: they are computed once per instance and kept in memory (ENUM_CACHE).
#   - t = 1: true cost c⊗(x0, b) (true_period_cost, fluid_lp_cg.jl), with κ^s ≤ k0^s. For a fixed
#     schedule, the best free days of pole s are its κ^s days of largest π_τ Cl^s_τ. The schedules
#     are enumerated once (ENUM_CACHE), the least cost of every (S, κ) is recomputed for every x0.
#   - Policy (Section 3.4): joint action of largest weight μ*_{b,1}.
#
# Requires fluid_lp_cg.jl and fluid_cg_paper.jl.

using JuMP, Gurobi, Printf
const MOI = JuMP.MOI

# ─────────────────────────────────────────────────────────────────────────────
# Enumeration of the joint actions
# ─────────────────────────────────────────────────────────────────────────────

struct JointActionEnum
    scheds::Vector{Vector{Int}}     # every admissible schedule: start day of every component, 0 if not maintained
    sched_set::Vector{Int}          # index in `sets` of the set of components maintained by each schedule
    sets::Vector{BitVector}         # every set of components that can be maintained in the same period
    kappas::Vector{Vector{Int}}     # every vector (κ^s)_s, κ^s ∈ qrange(K0^s)
    # t ≥ 2: one column per (set, κ vector), with its least cost f(b) and the schedule that attains it
    col_set::Vector{Int}
    col_kappa::Vector{Int}          # index in `kappas`
    col_sched::Vector{Int}
    col_cost::Vector{Float64}
    time::Float64                   # computing time of the enumeration
end

"Every admissible schedule: start days on the grid of the pricing problems, technician capacity on every day."
function enumerate_schedules(fd::FluidData)
    inst = fd.inst; M = fd.M; T = inst.T_period
    starts = [collect(FIRST_START_DAY:QUOTA_STEP:(T - fd.d[m] + 1 - inst.margin)) for m in 1:M]
    scheds = Vector{Vector{Int}}()
    load = zeros(T); cur = zeros(Int, M)
    function rec(m)
        if m > M
            push!(scheds, copy(cur)); return
        end
        rec(m + 1)   # m not maintained
        for s in starts[m]
            days = s:(s + fd.d[m] - 1)
            all(load[τ] + fd.rho[m] <= inst.R + 1e-9 for τ in days) || continue
            load[days] .+= fd.rho[m]; cur[m] = s
            rec(m + 1)
            load[days] .-= fd.rho[m]; cur[m] = 0
        end
    end
    rec(1)
    return scheds
end

"Penalty Σ_{τ ∉ Free} w_τ of the best set Free of κ days (its κ days of largest w_τ), for every κ of `κs`."
function best_free_costs(w::Vector{Float64}, κs)
    cs = cumsum(sort(w; rev = true)); tot = sum(w)
    return [κ == 0 ? tot : tot - cs[κ] for κ in κs]
end

"The κ days of largest w_τ (ties broken by the earliest day)."
best_free_days(w::Vector{Float64}, κ::Int) = sort(sortperm(w; rev = true, alg = Base.Sort.DEFAULT_STABLE)[1:κ])

"π_τ m^s_τ: penalty of the maintenance days of pole p (cost f(b) before the free days)."
function maintenance_weights(fd::FluidData, start::Vector{Int}, p::Int)
    T = fd.inst.T_period; pen = fd.inst.poles[p].daily_penalty
    w = zeros(T)
    for m in fd.comps_of_pole[p]
        start[m] > 0 || continue
        for τ in start[m]:(start[m] + fd.d[m] - 1); w[τ] = pen[τ]; end
    end
    return w
end

"π_τ Cl^s_τ: penalty of the capacity loss of pole p on every day (true cost before the free days)."
function loss_weights(fd::FluidData, x0::Vector{Int}, start::Vector{Int}, p::Int)
    T = fd.inst.T_period; pen = fd.inst.poles[p].daily_penalty; ms = fd.comps_of_pole[p]
    w = zeros(T)
    for τ in 1:T
        if any(start[m] > 0 && start[m] <= τ <= start[m] + fd.d[m] - 1 for m in ms)
            w[τ] = pen[τ]
        else
            w[τ] = pen[τ] * maximum(capacity_loss(fd, m, (start[m] > 0 && τ >= start[m] + fd.d[m]) ? 1 : x0[m]) for m in ms)
        end
    end
    return w
end

kappa_index(κ::Int) = κ ÷ QUOTA_STEP + 1

const ENUM_CACHE = Dict{Tuple{UInt,Int,Int},JointActionEnum}()

"Enumeration of the joint actions of `fd`, computed once and kept in memory (ENUM_CACHE)."
joint_action_enum(fd::FluidData) =
    get!(() -> build_joint_action_enum(fd), ENUM_CACHE, (objectid(fd), QUOTA_STEP, FIRST_START_DAY))

function build_joint_action_enum(fd::FluidData)
    t0 = time()
    K = fd.inst.n_poles
    scheds = enumerate_schedules(fd)
    set_id = Dict{BitVector,Int}(); sets = BitVector[]
    sched_set = [get!(() -> (push!(sets, s .> 0); length(sets)), set_id, s .> 0) for s in scheds]
    kappas = [collect(κ) for κ in Iterators.product((qrange(fd.K0[p]) for p in 1:K)...)][:]
    # least cost f(b) of every (set, κ vector) over its schedules
    best = fill(Inf, length(sets), length(kappas)); arg = zeros(Int, length(sets), length(kappas))
    for (i, s) in enumerate(scheds)
        mob = any(>(0), s) ? MOBILIZATION_COST : 0.0
        costs = [best_free_costs(maintenance_weights(fd, s, p), qrange(fd.K0[p])) for p in 1:K]
        for (j, κ) in enumerate(kappas)
            c = mob + sum(costs[p][kappa_index(κ[p])] for p in 1:K)
            if c < best[sched_set[i], j] - 1e-12
                best[sched_set[i], j] = c; arg[sched_set[i], j] = i
            end
        end
    end
    col_set = Int[]; col_kappa = Int[]; col_sched = Int[]; col_cost = Float64[]
    for si in eachindex(sets), j in eachindex(kappas)
        push!(col_set, si); push!(col_kappa, j); push!(col_sched, arg[si, j]); push!(col_cost, best[si, j])
    end
    return JointActionEnum(scheds, sched_set, sets, kappas, col_set, col_kappa, col_sched, col_cost, time() - t0)
end

"""
First-step columns b ∈ B(x0): every (set, κ vector) with κ^s ≤ k0^s, with its least true cost
c⊗(x0, b) over its schedules. Returns (set, κ index, schedule, cost) vectors.
"""
function first_step_enum(fd::FluidData, E::JointActionEnum, x0::Vector{Int}, k0::Vector{Int})
    K = fd.inst.n_poles
    feas = [j for (j, κ) in enumerate(E.kappas) if all(κ[p] <= k0[p] for p in 1:K)]
    best = fill(Inf, length(E.sets), length(feas)); arg = zeros(Int, length(E.sets), length(feas))
    for (i, s) in enumerate(E.scheds)
        repl = sum(fd.cc[m] for m in 1:fd.M if s[m] > 0; init = 0.0) + (any(>(0), s) ? MOBILIZATION_COST : 0.0)
        costs = [best_free_costs(loss_weights(fd, x0, s, p), qrange(fd.K0[p])) for p in 1:K]
        for (jj, j) in enumerate(feas)
            κ = E.kappas[j]
            c = repl + sum(costs[p][kappa_index(κ[p])] for p in 1:K)
            if c < best[E.sched_set[i], jj] - 1e-12
                best[E.sched_set[i], jj] = c; arg[E.sched_set[i], jj] = i
            end
        end
    end
    c_set = Int[]; c_kappa = Int[]; c_sched = Int[]; c_cost = Float64[]
    for si in eachindex(E.sets), (jj, j) in enumerate(feas)
        push!(c_set, si); push!(c_kappa, j); push!(c_sched, arg[si, jj]); push!(c_cost, best[si, jj])
    end
    return c_set, c_kappa, c_sched, c_cost
end

"Joint action (schedule and free days) of a first-step column: the free days are the κ^s days of largest π_τ Cl^s_τ."
function first_step_column(fd::FluidData, E::JointActionEnum, x0::Vector{Int}, sched::Int, j::Int)
    s = E.scheds[sched]; κ = E.kappas[j]
    return Column(copy(s), [best_free_days(loss_weights(fd, x0, s, p), κ[p]) for p in 1:fd.inst.n_poles])
end

# ─────────────────────────────────────────────────────────────────────────────
# Full fluid LP with every column
# ─────────────────────────────────────────────────────────────────────────────

"Add the constraint Σ_i coefs_i vars_i == rhs through MOI, without building a JuMP expression (rows with many terms)."
function add_linear_eq!(model::Model, vars::Vector{VariableRef}, coefs::Vector{Float64}, rhs::Float64)
    terms = [MOI.ScalarAffineTerm(c, index(v)) for (v, c) in zip(vars, coefs)]
    MOI.add_constraint(backend(model), MOI.ScalarAffineFunction(terms, 0.0), MOI.EqualTo(rhs))
end

"""
Solve the LP. It is always feasible, but Gurobi's presolve occasionally declares it infeasible (observed on two_poles
with one thread, as for the master LP of fluid_cg_paper.jl): it is then solved again without presolve, then without
presolve and with NumericFocus = 3.
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
    solve_fluid_enum_paper(fd, x0, k0, t0_cyc, H; lp_params, verbose)

Solves the Full fluid LP (eq:fluid) with all the joint actions of the enumeration (no column
generation) and returns the first-step joint action of largest weight μ1*_b (Section 3.4), with
statistics. Same LP as the order-1 master of fluid_cg_paper.jl (build_master_paper), with every
column.
"""
function solve_fluid_enum_paper(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, t0_cyc::Int, H::Int;
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

    # variables q^m_{(n,k),(mode,κ),t}, t ≥ 2, κ ≤ k (eq:fluid-admissible), as in build_master_paper
    q = Dict{NTuple{6,Int},VariableRef}()
    obj = AffExpr(0.0)
    for m in 1:M, t in 2:H, n in 1:fd.n_states[m], mode in 1:2
        for k in qrange(fd.K0[pole_of[m]]), κ in qrange(k)
            v = @variable(model, lower_bound = 0)
            q[(m, n, mode, k, κ, t)] = v
            add_to_expression!(obj, paper_local_cost(fd, m, n, mode, κ), v)
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

    # Σ_b μ1_b = 1 (eq:fluid-first)
    add_linear_eq!(model, μ1, ones(length(μ1)), 1.0)

    # initial condition (eq:fluid-initial): Σ_a q^m_{(n,k),a,2} = Σ_b p_m((n,k) | x0_m, A_m(b)) μ1_b
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
            qs = [q[(m, n, mode, k, κ, t)] for n in 1:fd.n_states[m] for k in κ:QUOTA_STEP:K0]
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
    solve_quota_subsystem_enum_paper(fd, x0, k0, t0_cyc, H; verbose)

Solves the second formulation of the Full fluid LP of Section 5.2 of paper.tex
(eq:fluid-quota-subsystem), with an additional subsystem per pole for its quota, with all the
joint actions of the enumeration. Variables q^m_{n,δ,t} of the components, without the quota,
with the cost c̃_m(n, δ) = δ Rc_m + T π ℓ_m(n), r^s_{k,κ,t} (κ ≤ k) of the quota subsystem of
pole s, without cost, and μ_{b,t}, μ1_b as in solve_fluid_enum_paper. Returns the first-step
joint action of largest weight μ1*_b (Section 3.4), with statistics.
"""
function solve_quota_subsystem_enum_paper(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, t0_cyc::Int, H::Int;
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

    add_linear_eq!(model, μ1, ones(length(μ1)), 1.0)   # Σ_b μ1_b = 1
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
                rs = [r[(s, k, κ, t)] for k in κ:QUOTA_STEP:fd.K0[s]]
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

"""
    solve_order2_enum_paper(fd, x0, k0, t0_cyc, H; pairs, lp_method, verbose)

Solves the order-2 fluid LP (Section 4.2) restricted to the groups of `pairs` with all the joint
actions of the enumeration (no column generation): the reduced master of fluid_cg_paper.jl
(build_master_paper with `reduced = true`, same LP as the order-2 column generation) with every
column. Barrier (`lp_method = 2`) by default: the dual simplex is very slow on this degenerate LP.
Returns the first-step joint action of largest weight μ1*_b (Section 3.4), with statistics.
"""
function solve_order2_enum_paper(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, t0_cyc::Int, H::Int;
                                 pairs::Vector{Tuple{Int,Int}} = same_pole_pairs(fd), lp_method::Int = 2,
                                 verbose::Bool = false)
    # same cleaning of the transitions as the order-2 column generation (ORDER2_P_TOL, fluid_cg_paper.jl)
    old_tol = PAPER_P_TOL
    old_tol == ORDER2_P_TOL || set_paper_p_tol!(ORDER2_P_TOL)
    try
        return _solve_order2_enum_paper(fd, x0, k0, t0_cyc, H, pairs, lp_method, verbose)
    finally
        old_tol == ORDER2_P_TOL || set_paper_p_tol!(old_tol)
    end
end

function _solve_order2_enum_paper(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, t0_cyc::Int, H::Int,
                                  pairs::Vector{Tuple{Int,Int}}, lp_method::Int, verbose::Bool)
    t_begin = time()
    E = joint_action_enum(fd)
    t_first = time()
    c_set, c_kappa, c_sched, c_cost = first_step_enum(fd, E, x0, k0)
    t_first = time() - t_first
    t_build = time()
    master = build_master_paper(fd, x0, k0, t0_cyc, H; pairs = pairs, lp_method = lp_method, reduced = true)
    for b in eachindex(c_cost)
        add_first_column!(fd, master, first_step_column(fd, E, x0, c_sched[b], c_kappa[b]))
    end
    if H >= 2
        for b in eachindex(E.col_cost)
            s = E.scheds[E.col_sched[b]]; κ = E.kappas[E.col_kappa[b]]
            add_column_paper!(fd, master, Column(copy(s), [best_free_days(maintenance_weights(fd, s, p), κ[p])
                                                          for p in 1:fd.inst.n_poles]))
        end
    end
    t_build = time() - t_build
    t_solve = time()
    solve_master!(master)
    t_solve = time() - t_solve
    b1 = argmax(value.(master.μ1))
    info = (time = time() - t_begin, lp_value = objective_value(master.model),
            t_enum = E.time, t_first = t_first, t_build = t_build, t_solve = t_solve,
            n_columns = length(master.μ1) + (H - 1) * length(master.cols),
            iterations = 0, converged = true)
    verbose && @printf("  order 2: first step %.2fs  build %.2fs  solve %.2fs  LP %.6f  columns %d\n",
                       t_first, t_build, t_solve, info.lp_value, info.n_columns)
    return master.cols1[b1], info
end

"""
Fluid policy (Section 3.4) of the Full fluid LP solved with every column, or of the order-2 fluid
LP if `pairs` is not empty: (x, k, t_global) -> (Column, solve_time). Solutions are cached (in
memory and in `cache_file`), as in make_policy_fluid_cg_paper.
"""
function make_policy_fluid_enum_paper(fd::FluidData; H::Int = 6,
                                      pairs::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[],
                                      quota_subsystem::Bool = false,
                                      cache_file::Union{String,Nothing} = nothing,
                                      log_solve::Function = info -> nothing)
    T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
    cache, next_cache_id = load_cache(cache_file)

    function policy(x::Vector{Int}, k::Vector{Int}, t_global::Int)
        t0_cyc = (t_global - 1) % N_PERIODS_PER_YEAR + 1
        H_eff = min(H, T_total - t_global + 1)
        key = (Tuple(x), Tuple(k), t0_cyc, H_eff)
        if !haskey(cache, key)
            col, info = quota_subsystem ? solve_quota_subsystem_enum_paper(fd, x, k, t0_cyc, H_eff) :
                        isempty(pairs) ? solve_fluid_enum_paper(fd, x, k, t0_cyc, H_eff) :
                                         solve_order2_enum_paper(fd, x, k, t0_cyc, H_eff; pairs = pairs)
            cache[key] = (col.start, col.free_days, info.time, info.lp_value,
                          info.iterations, info.n_columns, info.converged)
            log_solve((key = key, time = info.time, lp_value = info.lp_value,
                       iterations = info.iterations, n_columns = info.n_columns,
                       converged = info.converged, t_milp = 0.0, t_lp = info.t_solve))
            append_cache_entry!(cache_file, next_cache_id, key, cache[key])
            return col, info.time
        end
        start, free_days = cache[key][1], cache[key][2]
        return Column(start, free_days), 0.0
    end
    return policy
end
