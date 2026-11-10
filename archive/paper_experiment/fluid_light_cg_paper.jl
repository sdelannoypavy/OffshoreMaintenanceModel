# fluid_light_cg_paper.jl
#
# Light fluid LP of paper.tex (eq:fluid-quota-subsystem, the LP of solve_quota_subsystem_enum_paper in
# fluid_enum_paper.jl), solved by column generation as the Full fluid LP in fluid_cg_paper.jl.
#
#   - Variables q^m_{n,δ,t}, t ≥ 2, of the components, without the quota, with the cost
#     c̃_m(n, δ) = δ Rc_m + T π ℓ_m(n), r^s_{k,κ,t} (κ ≤ k) of the quota subsystem of pole s, without
#     cost, μ1_b for b ∈ B'_1 ⊆ B(x0) with the true cost c⊗(x0, b), and μ_{b,t} for b ∈ B'_t ⊆ B with
#     the cost f(b) = Mc, as in solve_quota_subsystem_enum_paper.
#   - A column b enters the rows of the components through its modes A_m(b) = δ only, and the rows of
#     the quotas through its numbers of declared days κ^s(b) only. The pricing problems are those of
#     fluid_cg_paper.jl (build_pricing_core), with the duals written with the indicators z_m of the
#     maintained components and ksel[s, κ] of the numbers of declared days.
#   - Policy (Section 3.4): joint action of largest weight μ*_{b,1}.
#
# Requires fluid_lp_cg.jl and fluid_cg_paper.jl.

using JuMP, Gurobi, Printf

mutable struct LightMaster
    model::Model
    H::Int
    x0::Vector{Int}
    k0::Vector{Int}
    t0_cyc::Int
    first::ConstraintRef                          # Σ_b μ1_b = 1
    cinit::Dict{Tuple{Int,Int},ConstraintRef}     # (m, n) -> initial condition of the component at t = 2
    qinit::Dict{Tuple{Int,Int},ConstraintRef}     # (s, k) -> initial condition of the quota at t = 2
    ccons::Dict{NTuple{3,Int},ConstraintRef}      # (m, δ, t) -> consistency of the component, t ≥ 2
    qcons::Dict{NTuple{3,Int},ConstraintRef}      # (s, κ, t) -> consistency of the quota, t ≥ 2
    μ1::Vector{VariableRef}
    cols1::Vector{Column}
    μ::Vector{Vector{VariableRef}}                # μ[col_id][t - 1], t = 2..H
    cols::Vector{Column}
    lp_method::Int
end

"Restricted master LP of the Light fluid LP for the initial state (x0, k0), without any column."
function build_master_light(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, t0_cyc::Int, H::Int; lp_method::Int = 1)
    M = fd.M; K = fd.inst.n_poles; T = fd.inst.T_period; π = constant_daily_penalty(fd)
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    set_optimizer_attribute(model, "Method", lp_method)
    q = Dict{NTuple{4,Int},VariableRef}()
    obj = AffExpr(0.0)
    for m in 1:M, t in 2:H, n in 1:fd.n_states[m], mode in 1:2
        v = @variable(model, lower_bound = 0)
        q[(m, n, mode, t)] = v
        add_to_expression!(obj, (mode == 2 ? fd.cc[m] : 0.0) + T * π * capacity_loss(fd, m, n), v)
    end
    r = Dict{NTuple{4,Int},VariableRef}()
    for s in 1:K, t in 2:H, k in qrange(fd.K0[s]), κ in qrange(k)
        r[(s, k, κ, t)] = @variable(model, lower_bound = 0)
    end
    first = @constraint(model, AffExpr(0.0) == 1.0)
    cinit = Dict{Tuple{Int,Int},ConstraintRef}(); qinit = Dict{Tuple{Int,Int},ConstraintRef}()
    if H >= 2
        for m in 1:M, n in 1:fd.n_states[m]
            cinit[(m, n)] = @constraint(model, q[(m, n, 1, 2)] + q[(m, n, 2, 2)] == 0)
        end
        for s in 1:K, k in qrange(fd.K0[s])
            qinit[(s, k)] = @constraint(model, sum(r[(s, k, κ, 2)] for κ in qrange(k)) == 0)
        end
    end
    # flows of the components and of the quotas, t ≥ 3
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
    # consistency of the components and of the quotas, t ≥ 2 (the columns are added by add_column_light!)
    ccons = Dict{NTuple{3,Int},ConstraintRef}(); qcons = Dict{NTuple{3,Int},ConstraintRef}()
    for t in 2:H
        for m in 1:M, mode in 1:2
            ccons[(m, mode, t)] = @constraint(model, sum(q[(m, n, mode, t)] for n in 1:fd.n_states[m]) == 0)
        end
        for s in 1:K, κ in qrange(fd.K0[s])
            qcons[(s, κ, t)] = @constraint(model, sum(r[(s, k, κ, t)] for k in κ:QUOTA_STEP:fd.K0[s]) == 0)
        end
    end
    @objective(model, Min, obj)
    return LightMaster(model, H, copy(x0), copy(k0), t0_cyc, first, cinit, qinit, ccons, qcons,
                       VariableRef[], Column[], Vector{VariableRef}[], Column[], lp_method)
end

"Add the variable μ1_b of a first-step column b ∈ B(x0), with its true cost c⊗(x0, b)."
function add_first_column_light!(fd::FluidData, master::LightMaster, col::Column)
    v = @variable(master.model, lower_bound = 0)
    set_objective_coefficient(master.model, v, first_step_cost(fd, master.x0, col))
    set_normalized_coefficient(master.first, v, 1.0)
    if master.H >= 2
        for m in 1:fd.M, n in 1:fd.n_states[m]
            prob = trans_prob(fd, m, n, master.x0[m], column_mode(col, m))
            prob == 0.0 || set_normalized_coefficient(master.cinit[(m, n)], v, -prob)
        end
        for s in 1:fd.inst.n_poles
            k_next = quota_next(1, master.t0_cyc, master.k0[s], column_kappa(col, s), fd.K0[s])
            set_normalized_coefficient(master.qinit[(s, k_next)], v, -1.0)
        end
    end
    push!(master.μ1, v); push!(master.cols1, col)
end

"Add the variables μ_{b,t}, t = 2..H, of a column b ∈ B, with the joint action cost f(b)."
function add_column_light!(fd::FluidData, master::LightMaster, col::Column)
    f = column_penalty(fd, col)
    vars = VariableRef[]
    for t in 2:master.H
        v = @variable(master.model, lower_bound = 0)
        set_objective_coefficient(master.model, v, f)
        for m in 1:fd.M
            set_normalized_coefficient(master.ccons[(m, column_mode(col, m), t)], v, -1.0)
        end
        for s in 1:fd.inst.n_poles
            set_normalized_coefficient(master.qcons[(s, column_kappa(col, s), t)], v, -1.0)
        end
        push!(vars, v)
    end
    push!(master.μ, vars); push!(master.cols, col)
end

"Indicator of A_m(b) = δ (mode only), as an affine expression of the pricing variables."
mode_ind(core::PricingCore, m::Int, mode::Int) = mode == 2 ? 1.0 * core.z[m] : 1.0 - core.z[m]

"First-step pricing problem of the Light fluid LP: min over b ∈ B(x0) of the reduced cost of μ1_b."
function solve_pricing_first_light(fd::FluidData, master::LightMaster, η::Float64, ν::Dict, νq::Dict;
                                   time_limit::Float64 = 60.0, n_pool::Int = 1, tol::Float64 = 1e-6)
    x0 = master.x0; k0 = master.k0
    core = build_pricing_core(fd; kcap = k0, x0 = x0, time_limit = time_limit, n_pool = n_pool)
    if master.H >= 2
        for m in 1:fd.M, mode in 1:2
            coef = sum(ν[(m, n)] * trans_prob(fd, m, n, x0[m], mode) for n in 1:fd.n_states[m])
            abs(coef) > 1e-12 && add_to_expression!(core.obj, coef, mode_ind(core, m, mode))
        end
        for s in 1:fd.inst.n_poles, κ in qrange(k0[s])
            coef = νq[(s, quota_next(1, master.t0_cyc, k0[s], κ, fd.K0[s]))]
            abs(coef) > 1e-12 && add_to_expression!(core.obj, coef, core.ksel[(s, κ)])
        end
    end
    col, val, extras = solve_core!(core, fd; pool_tol = η - tol)
    return col, val - η, extras
end

"Pricing problem of the Light fluid LP at stage t ≥ 2: min over b ∈ B of the reduced cost of μ_{b,t}."
function solve_pricing_t_light(fd::FluidData, λ::Dict, λq::Dict, t::Int;
                               time_limit::Float64 = 60.0, n_pool::Int = 1, tol::Float64 = 1e-6)
    core = build_pricing_core(fd; time_limit = time_limit, n_pool = n_pool)
    for m in 1:fd.M, mode in 1:2
        λv = λ[(m, mode, t)]
        abs(λv) > 1e-12 && add_to_expression!(core.obj, λv, mode_ind(core, m, mode))
    end
    for s in 1:fd.inst.n_poles, κ in qrange(fd.K0[s])
        λv = λq[(s, κ, t)]
        abs(λv) > 1e-12 && add_to_expression!(core.obj, λv, core.ksel[(s, κ)])
    end
    return solve_core!(core, fd; pool_tol = -tol)
end

"""
    solve_light_cg_paper(fd, x0, k0, t0_cyc, H; pool1, pool, ...)

Solves the Light fluid LP by column generation (Algorithm 1) and returns the first-step joint action
of largest weight μ1*_b (Section 3.4), with statistics. `pool1` and `pool` are columns found for
other states, used to initialise B'_1 (those that satisfy the quota constraint in k0) and B'_t.
"""
function solve_light_cg_paper(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, t0_cyc::Int, H::Int;
                              pool1::Vector{Column} = Column[], pool::Vector{Column} = Column[],
                              max_iter::Int = 50, tol::Float64 = 1e-6,
                              time_limit_pricing::Float64 = 60.0, n_pool::Int = 10, verbose::Bool = false)
    t_begin = time()
    master = build_master_light(fd, x0, k0, t0_cyc, H)
    feasible_first(col) = all(column_kappa(col, p) <= k0[p] for p in 1:fd.inst.n_poles)
    push_first!(col) = (col in master.cols1 || !feasible_first(col)) ? false : (add_first_column_light!(fd, master, col); true)
    push_col!(col) = col in master.cols ? false : (add_column_light!(fd, master, col); true)

    # Initial columns: the empty schedule (always in B(x0)), the singletons, and the columns of the pools.
    push_first!(empty_column(fd)); foreach(push_first!, pool1)
    if H >= 2
        push_col!(empty_column(fd))
        for m in 1:fd.M
            FIRST_START_DAY + fd.d[m] - 1 <= fd.inst.T_period && push_col!(singleton_column(fd, m, FIRST_START_DAY))
        end
        foreach(push_col!, pool)
    end

    iterations = 0; converged = false; t_master = 0.0; t_pricing = 0.0
    for iter in 1:max_iter
        iterations += 1
        t_m = time(); solve_master!(master); t_m = time() - t_m; t_master += t_m
        lpv = objective_value(master.model)
        # every dual is read before any column is added
        η = dual(master.first)
        ν = Dict(k => dual(c) for (k, c) in master.cinit); νq = Dict(k => dual(c) for (k, c) in master.qinit)
        λ = Dict(k => dual(c) for (k, c) in master.ccons); λq = Dict(k => dual(c) for (k, c) in master.qcons)
        n_new = 0
        t_p = time()
        col, rc, extras = solve_pricing_first_light(fd, master, η, ν, νq; time_limit = time_limit_pricing,
                                                    n_pool = n_pool, tol = tol)
        if !isnothing(col) && rc < -tol
            push_first!(col) && (n_new += 1)
            foreach(c -> push_first!(c) && (n_new += 1), extras)
        end
        for t in 2:H
            col, rc, extras = solve_pricing_t_light(fd, λ, λq, t; time_limit = time_limit_pricing, n_pool = n_pool, tol = tol)
            (isnothing(col) || rc >= -tol) && continue
            push_col!(col) && (n_new += 1)
            foreach(c -> push_col!(c) && (n_new += 1), extras)
        end
        t_pricing += time() - t_p
        if verbose
            @printf("    iter %3d  master %.2fs  pricing %.2fs  new %3d  LP %.6f\n", iterations, t_m, time() - t_p, n_new, lpv)
            flush(stdout)
        end
        if n_new == 0
            converged = true
            break
        end
    end
    if !converged
        t_m = time(); solve_master!(master); t_master += time() - t_m
    end

    weights = value.(master.μ1)
    best = master.cols1[argmax(weights)]
    return best, (time = time() - t_begin, lp_value = objective_value(master.model),
                  t_master = t_master, t_pricing = t_pricing, iterations = iterations, converged = converged,
                  n_columns = length(master.cols1) + length(master.cols), cols1 = master.cols1, cols = master.cols)
end

"""
Fluid policy (Section 3.4) of the Light fluid LP solved by column generation: (x, k, t_global) ->
(Column, solve_time). Solutions are cached (in memory and in `cache_file`), and the columns found for
a state are kept in pools per (t0_cyc, H_eff), as in make_policy_fluid_cg_paper.
"""
function make_policy_light_cg_paper(fd::FluidData; H::Int = 6,
                                    cache_file::Union{String,Nothing} = nothing,
                                    log_solve::Function = info -> nothing,
                                    max_iter::Int = 50, time_limit_pricing::Float64 = 60.0)
    T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
    cache, next_cache_id = load_cache(cache_file)
    pools1 = Dict{Tuple{Int,Int},Vector{Column}}()
    pools = Dict{Tuple{Int,Int},Vector{Column}}()

    function policy(x::Vector{Int}, k::Vector{Int}, t_global::Int)
        t0_cyc = (t_global - 1) % N_PERIODS_PER_YEAR + 1
        H_eff = min(H, T_total - t_global + 1)
        key = (Tuple(x), Tuple(k), t0_cyc, H_eff)
        if !haskey(cache, key)
            pk = (t0_cyc, H_eff)
            col, info = solve_light_cg_paper(fd, x, k, t0_cyc, H_eff; pool1 = get(pools1, pk, Column[]),
                                             pool = get(pools, pk, Column[]), max_iter = max_iter,
                                             time_limit_pricing = time_limit_pricing)
            pools1[pk] = unique(vcat(get(pools1, pk, Column[]), info.cols1))
            pools[pk] = unique(vcat(get(pools, pk, Column[]), info.cols))
            cache[key] = (col.start, col.free_days, info.time, info.lp_value,
                          info.iterations, info.n_columns, info.converged)
            log_solve((key = key, time = info.time, lp_value = info.lp_value,
                       iterations = info.iterations, n_columns = info.n_columns,
                       converged = info.converged, t_milp = info.t_pricing, t_lp = info.t_master))
            append_cache_entry!(cache_file, next_cache_id, key, cache[key])
            return col, info.time
        end
        start, free_days = cache[key][1], cache[key][2]
        return Column(start, free_days), 0.0
    end
    return policy
end
