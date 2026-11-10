# fluid_cg_light.jl
#
# A lighter fluid LP (Section 3.2's Full Fluid LP, but without the quota-augmented local
# state of Section 5.3): instead of giving every component of a pole its own copy of the
# pole's remaining free-day quota k (so that q is indexed by (x_m, k, kappa)), we introduce
# one extra subsystem per pole, m = 0 say, whose local state is k and whose local action is
# kappa (the number of free days used this period, kappa <= k). Real components m = 1..M
# then only carry their own degradation state x_m and mode a_m: q^m_{x_m,a_m,t} loses the
# (k, kappa) indices entirely.
#
# This drops the exact correlation between a component's own state and the pole's quota
# level that the quota-augmented design tracks (see the discussion in Section 5.3 of the
# paper: "we expect maintenance decisions for a given component to depend on k a lot"), so
# fluid_cg_light gives a looser LP relaxation than fluid_cg -- but with a much smaller
# master LP (no more (k, kappa) blow-up of every component's index, no disaggregated
# mu^{(p)}_{k,b,t} variables) and a much simpler pricing MILP (the duals no longer couple a
# component's mode to the pole's (k, kappa) pair, so the reduced cost is linear in z_m and
# in the quota selection separately -- no McCormick linearisation is needed at all).
#
# Requires fluid_lp_cg.jl to be loaded first (provides FluidData, Column, stage_cyc,
# empty_column, singleton_column, column_mode, column_kappa, column_penalty, GRB_ENV, ...).

using JuMP, Printf

# ─────────────────────────────────────────────────────────────────────────────
# Restricted master LP
# ─────────────────────────────────────────────────────────────────────────────

mutable struct MasterLPLight
    model::Model
    H::Int
    consm::Dict{NTuple{3,Int},ConstraintRef}   # (m, mode, t)
    consκ::Dict{NTuple{3,Int},ConstraintRef}   # (pole, κ, t)
    q::Dict{NTuple{4,Int},VariableRef}         # (m, x, mode, t)
    qk::Dict{NTuple{4,Int},VariableRef}        # (pole, k, κ, t)
    μ::Vector{Vector{VariableRef}}             # μ[col_id][t]
end

"""
Builds the light Fluid LP: real components m = 1..M, indexed only by their own (state,
mode), plus one quota subsystem per pole, indexed by (k, κ). Same Method = 1 (dual simplex)
default as build_master (see its docstring for why).
"""
function build_master_light(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, t0_cyc::Int, H::Int)
    M = fd.M; K = fd.inst.n_poles
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    set_optimizer_attribute(model, "Method", 1)

    q = Dict{NTuple{4,Int},VariableRef}()
    obj = AffExpr(0.0)
    for m in 1:M, t in 1:H, mode in 1:2, x in 1:fd.n_states[m]
        v = @variable(model, lower_bound = 0)
        q[(m, x, mode, t)] = v
        add_to_expression!(obj, local_cost(fd, m, x, mode), v)
    end

    qk = Dict{NTuple{4,Int},VariableRef}()
    for p in 1:K, t in 1:H, k in qrange(fd.K0[p]), κ in qrange(k)
        qk[(p, k, κ, t)] = @variable(model, lower_bound = 0)
    end
    @objective(model, Min, obj)

    # ── Initial conditions ───────────────────────────────────────────────────
    for m in 1:M, x in 1:fd.n_states[m]
        @constraint(model, sum(q[(m, x, mode, 1)] for mode in 1:2) == (x == x0[m] ? 1.0 : 0.0))
    end
    for p in 1:K, k in qrange(fd.K0[p])
        @constraint(model, sum(qk[(p, k, κ, 1)] for κ in qrange(k)) == (k == k0[p] ? 1.0 : 0.0))
    end

    # ── Flow: components (ordinary single-component MDP, no quota entanglement) ──
    for t in 2:H, m in 1:M
        P = fd.P[m]
        for x in 1:fd.n_states[m]
            lhs = sum(q[(m, x, mode, t)] for mode in 1:2)
            rhs = AffExpr(0.0)
            for xp in 1:fd.n_states[m]
                prob = P[x, xp]
                prob > 1e-12 && add_to_expression!(rhs, prob, q[(m, xp, 1, t - 1)])
            end
            x == 1 && for xp in 1:fd.n_states[m]
                add_to_expression!(rhs, 1.0, q[(m, xp, 2, t - 1)])
            end
            @constraint(model, lhs == rhs)
        end
    end

    # ── Flow: quota subsystem of each pole (state k, action κ <= k, deterministic
    #    transition k' = k - κ, reset to K0 every year) ──────────────────────────
    for t in 2:H, p in 1:K
        K0 = fd.K0[p]
        reset = stage_cyc(t - 1, t0_cyc) == N_PERIODS_PER_YEAR
        for k in qrange(K0)
            lhs = sum(qk[(p, k, κ, t)] for κ in qrange(k))
            if reset
                if k != K0
                    @constraint(model, lhs == 0)
                    continue
                end
                rhs = sum(qk[(p, kp, κp, t - 1)] for kp in qrange(K0) for κp in qrange(kp))
            else
                rhs = sum(qk[(p, k + κp, κp, t - 1)] for κp in 0:QUOTA_STEP:(K0 - k))
            end
            @constraint(model, lhs == rhs)
        end
    end

    # ── Consistency with the joint action: each real component to its own mode,
    #    each pole's quota subsystem to its own κ. No disaggregation is needed here
    #    (unlike build_master's μ^{(p)}_{k,b,t}), since q no longer carries k. ────
    consm = Dict{NTuple{3,Int},ConstraintRef}()
    for t in 1:H, m in 1:M, mode in 1:2
        consm[(m, mode, t)] = @constraint(model, sum(q[(m, x, mode, t)] for x in 1:fd.n_states[m]) == 0)
    end
    consκ = Dict{NTuple{3,Int},ConstraintRef}()
    for t in 1:H, p in 1:K, κ in qrange(fd.K0[p])
        consκ[(p, κ, t)] = @constraint(model, sum(qk[(p, k, κ, t)] for k in κ:QUOTA_STEP:fd.K0[p]) == 0)
    end

    return MasterLPLight(model, H, consm, consκ, q, qk, Vector{VariableRef}[])
end

"Adds μ_{b,t} for every t, tying it to the (mode) consistency of every real component and to
the (κ) consistency of every pole's quota subsystem -- no per-k disaggregation required."
function add_column_light!(fd::FluidData, master::MasterLPLight, col::Column)
    f = column_penalty(fd, col)
    vars = Vector{VariableRef}(undef, master.H)
    for t in 1:master.H
        v = @variable(master.model, lower_bound = 0)
        set_objective_coefficient(master.model, v, f)
        for m in 1:fd.M
            set_normalized_coefficient(master.consm[(m, column_mode(col, m), t)], v, -1.0)
        end
        for p in 1:fd.inst.n_poles
            set_normalized_coefficient(master.consκ[(p, column_kappa(col, p), t)], v, -1.0)
        end
        vars[t] = v
    end
    push!(master.μ, vars)
end

"Same Method = 1 primary / Method = 2 fallback as solve_master!(::MasterLP)."
function solve_master!(master::MasterLPLight)
    set_optimizer_attribute(master.model, "Method", 1)
    optimize!(master.model)
    if !is_solved_and_feasible(master.model; dual = true)
        set_optimizer_attribute(master.model, "Method", 2)
        optimize!(master.model)
        assert_is_solved_and_feasible(master.model; dual = true)
    end
end

"Duals λ[m, mode, t] (components) and λκ[pole, κ+1, t] (quota subsystems)."
function consistency_duals_light(fd::FluidData, master::MasterLPLight)
    λ = zeros(fd.M, 2, master.H)
    for ((m, mode, t), con) in master.consm
        λ[m, mode, t] = dual(con)
    end
    maxK = maximum(fd.K0)
    λκ = zeros(fd.inst.n_poles, maxK + 1, master.H)
    for ((p, κ, t), con) in master.consκ
        λκ[p, κ + 1, t] = dual(con)
    end
    return λ, λκ
end

# ─────────────────────────────────────────────────────────────────────────────
# Pricing problem
#
# min_b  f(b) + Σ_m λ_t^{m, A_m(b)} + Σ_p λκ_t^{p, κ_p(b)}
#
# Unlike solve_pricing (fluid_lp_cg.jl), the duals no longer depend on (k, κ) jointly: a
# component's dual only depends on its own mode, and a pole's quota dual only depends on κ.
# The reduced cost is therefore already linear in z_m and in the κ-selection separately, so
# no McCormick linearisation (the y[m, k, κ] of solve_pricing) is needed.
# ─────────────────────────────────────────────────────────────────────────────

function solve_pricing_light(fd::FluidData, λ_t::Array{Float64,2}, λκ_t::Array{Float64,2};
                             time_limit::Float64 = 60.0)
    inst = fd.inst; M = fd.M; T = inst.T_period; K = inst.n_poles

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    set_time_limit_sec(model, time_limit)

    last_start = [T - fd.d[m] + 1 - inst.margin for m in 1:M]
    first_start = FIRST_START_DAY
    # Start days restricted to the grid first_start, first_start+QUOTA_STEP, ... (see
    # solve_pricing in fluid_lp_cg.jl for why this loses no optimal solution given the
    # current uniform daily penalty and durations that are multiples of QUOTA_STEP).
    starts_grid(m) = first_start:QUOTA_STEP:last_start[m]
    @variable(model, z[1:M], Bin)
    ms = Dict{Tuple{Int,Int},VariableRef}()
    for m in 1:M
        if last_start[m] < first_start
            fix(z[m], 0.0; force = true)
            continue
        end
        for τ in starts_grid(m)
            ms[(m, τ)] = @variable(model, binary = true)
        end
        @constraint(model, sum(ms[(m, τ)] for τ in starts_grid(m)) == z[m])
    end
    ong(m, τ) = sum((ms[(m, s)] for s in max(first_start, τ - fd.d[m] + 1):min(τ, last_start[m])
                     if haskey(ms, (m, s))); init = AffExpr(0.0))

    @variable(model, 0 <= any_m[1:K, 1:T] <= 1)
    @variable(model, free[1:K, 1:T], Bin)
    @variable(model, 0 <= w[1:K, 1:T] <= 1)

    # κsel[p, κ] = 1 iff pole p declares exactly κ free days this period (no k dimension)
    κsel = Dict{Tuple{Int,Int},VariableRef}()
    for p in 1:K, κ in qrange(fd.K0[p])
        κsel[(p, κ)] = @variable(model, binary = true)
    end

    for τ in first_start:QUOTA_STEP:T
        @constraint(model, sum(fd.rho[m] * ong(m, τ) for m in 1:M) <= inst.R)
    end
    for τ in 1:T
        for p in 1:K
            for m in fd.comps_of_pole[p]
                @constraint(model, any_m[p, τ] >= ong(m, τ))
            end
            @constraint(model, w[p, τ] <= free[p, τ])
            @constraint(model, w[p, τ] <= any_m[p, τ])
        end
    end
    for p in 1:K
        @constraint(model, sum(κsel[(p, κ)] for κ in qrange(fd.K0[p])) == 1)
        @constraint(model, sum(free[p, τ] for τ in 1:T) == sum(κ * κsel[(p, κ)] for κ in qrange(fd.K0[p])))
    end

    obj = AffExpr(0.0)
    for p in 1:K, τ in 1:T
        pen = inst.poles[p].daily_penalty[τ]
        add_to_expression!(obj, pen, any_m[p, τ])
        add_to_expression!(obj, -pen, w[p, τ])
    end
    for m in 1:M
        add_to_expression!(obj, λ_t[m, 1])
        add_to_expression!(obj, λ_t[m, 2] - λ_t[m, 1], z[m])
    end
    for p in 1:K, κ in qrange(fd.K0[p])
        add_to_expression!(obj, λκ_t[p, κ + 1], κsel[(p, κ)])
    end
    @objective(model, Min, obj)
    optimize!(model)

    status = termination_status(model)
    if !(status == MOI.OPTIMAL || (status == MOI.TIME_LIMIT && has_values(model)))
        return nothing, Inf
    end

    start = zeros(Int, M)
    for m in 1:M
        value(z[m]) > 0.5 || continue
        for τ in starts_grid(m)
            value(ms[(m, τ)]) > 0.5 && (start[m] = τ)
        end
    end
    free_days = [[τ for τ in 1:T if value(free[p, τ]) > 0.5] for p in 1:K]
    return Column(start, free_days), objective_value(model)
end

# ─────────────────────────────────────────────────────────────────────────────
# Column generation
# ─────────────────────────────────────────────────────────────────────────────

"Same interface as solve_fluid_cg (fluid_lp_cg.jl), using the light master/pricing."
function solve_fluid_cg_light(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, t0_cyc::Int, H::Int;
                              max_iter::Int = 50, tol::Float64 = 1e-6,
                              time_limit_pricing::Float64 = 60.0,
                              warm_start_cols::Vector{Column} = Column[],
                              heuristic_extraction::Bool = true)
    t_begin = time()
    master = build_master_light(fd, x0, k0, t0_cyc, H)
    cols = Column[]
    function push_col!(col)
        col in cols && return
        push!(cols, col); add_column_light!(fd, master, col)
    end
    push_col!(empty_column(fd))
    for m in 1:fd.M; push_col!(singleton_column(fd, m, 1)); end
    foreach(push_col!, warm_start_cols)

    iterations = 0
    converged = false
    for iter in 1:max_iter
        iterations = iter
        solve_master!(master)
        λ, λκ = consistency_duals_light(fd, master)
        n_new = 0
        for t in 1:H
            col, rc = solve_pricing_light(fd, λ[:, :, t], λκ[:, :, t]; time_limit = time_limit_pricing)
            (isnothing(col) || rc >= -tol || col in cols) && continue
            push_col!(col); n_new += 1
        end
        if n_new == 0
            converged = true
            break
        end
    end
    solve_master!(master)

    weights = [value(master.μ[i][1]) for i in eachindex(cols)]
    best = cols[argmax(weights)]
    if heuristic_extraction
        mode_mass(m, mode) = value(master.q[(m, x0[m], mode, 1)])
        kappa_mass(p, κ) = value(master.qk[(p, k0[p], κ, 1)])
        best = extract_policy_heuristic(fd, x0, k0, mode_mass, kappa_mass)
    end
    lp_value = objective_value(master.model)
    elapsed = time() - t_begin
    return best, (time = elapsed, lp_value = lp_value, iterations = iterations, converged = converged,
                  n_columns = length(cols), cols = cols)
end

# ─────────────────────────────────────────────────────────────────────────────
# Fluid policy with cache (same interface as make_policy_fluid_cg)
# ─────────────────────────────────────────────────────────────────────────────

function make_policy_fluid_cg_light(fd::FluidData; H::Int = 6, warm_start::Bool = true,
                                    cache_file::Union{String,Nothing} = nothing,
                                    log_solve::Function = info -> nothing,
                                    max_iter::Int = 50, time_limit_pricing::Float64 = 60.0,
                                    heuristic_extraction::Bool = true)
    T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
    cache, next_cache_id = load_cache(cache_file)
    last_cols = Column[]

    function policy(x::Vector{Int}, k::Vector{Int}, t_global::Int)
        t0_cyc = (t_global - 1) % N_PERIODS_PER_YEAR + 1
        H_eff = min(H, T_total - t_global + 1)
        key = (Tuple(x), Tuple(k), t0_cyc, H_eff)
        if !haskey(cache, key)
            col, info = solve_fluid_cg_light(fd, x, k, t0_cyc, H_eff; max_iter = max_iter,
                                             time_limit_pricing = time_limit_pricing,
                                             warm_start_cols = warm_start ? last_cols : Column[],
                                             heuristic_extraction = heuristic_extraction)
            last_cols = info.cols
            cache[key] = (col.start, col.free_days, info.time, info.lp_value,
                          info.iterations, info.n_columns, info.converged)
            log_solve((key = key, time = info.time, lp_value = info.lp_value,
                       iterations = info.iterations, n_columns = info.n_columns,
                       converged = info.converged))
            append_cache_entry!(cache_file, next_cache_id, key, cache[key])
            return col, info.time
        end
        start, free_days = cache[key][1], cache[key][2]
        return Column(start, free_days), 0.0
    end
    return policy
end
