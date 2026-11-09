# compact_lp.jl
#
# Compact Fluid LP (Section 3.3, Eq. (6); specialised to this instance in Appendix C.4).
# A single monolithic LP that avoids column generation altogether by relaxing the
# integrality constraints defining the joint action set B and embedding one continuous
# copy of the resulting polytope per period directly in the LP (Section 3.3). This gives
# a looser bound than the Full Fluid LP solved by column generation (fluid_lp_cg.jl),
# at the benefit of being solved as a single LP with no pricing loop.
#
# Adapted from column_generation/direct_lp.jl (single-pole prototype) to the multi-pole
# FluidData / Column structures of instances.jl and fluid_lp_cg.jl.
#
# Requires fluid_lp_cg.jl to be loaded first (provides FluidData, Column, stage_cyc,
# empty_column, GRB_ENV, ...).
#
# State-action variables q[m,x,mode,k,κ,t]: identical to the master LP of fluid_lp_cg.jl
# (mode = 1: not maintained, mode = 2: maintained; k = quota remaining at the start of the
# period, κ = free days used during the period, on the pole of m).
#
# Relaxed schedule variables per period t (relaxation of the MILP of Appendix C.1, Eqs.
# (10)-(12)), one copy per period since there is no weather scenario:
#   z[m,t]         in [0,1] : probability that component m is maintained during period t
#   m_start[m,τ,t] in [0,1] : probability that maintenance of m starts on day τ
#   any_m[p,τ,t]   in [0,1] : probability that some component of pole p is under maintenance
#                             on day τ
#   free[p,τ,t]    in [0,1] : probability that day τ of pole p is declared penalty-free
#   w[p,τ,t]       in [0,1] : linearisation of any_m ∧ free
#   ksel[p,κ,t]    in [0,1] : probability that pole p declares exactly κ free days
#
# Coupling constraints (Appendix C.4):
#   Σ_{x,k,κ} q[m,x,2,k,κ,t] <= z[m,t]                    maintenance action
#   Σ_τ free[p,τ,t] == Σ_{x,mode,k,κ} κ q[m1,x,mode,k,κ,t]  free-day action (m1 = reference
#     component of pole p; every component of a pole shares the same κ, Section 5.3)
#
# Quota dynamics (the k component of the local state) are embedded in the flow constraint,
# exactly as in fluid_lp_cg.jl; there is no separate global quota constraint.
#
# Warm start across states (make_policy_compact_lp): unlike fluid_lp_cg.jl's master, which is
# rebuilt from scratch for every new (x, k, t0_cyc, H) but warm-started through its reused
# COLUMNS, the compact LP has no such combinatorial objects to carry over. Its structure
# (variables, constraints, coefficients) depends only on (t0_cyc, H) -- x0 and k0 only ever
# enter through the initial-condition constraint's right-hand side and through which q
# variables are fixed to 0 at t=1 (see build_compact_template/configure_compact_LP! below), so
# the SAME model can be reused for every state sharing a given (t0_cyc, H): only a handful of
# bounds and right-hand sides are changed between solves, which lets Gurobi's simplex warm
# start from the previous basis instead of resolving from scratch (Method=2/barrier, used
# before, has no such warm start).

using JuMP, Gurobi, Printf

@isdefined(GRB_ENV) || (GRB_ENV = Gurobi.Env())

# ─────────────────────────────────────────────────────────────────────────────
# LP builder
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_compact_template(fd, t0_cyc, H)

Builds the compact LP for a given `(t0_cyc, H)`, generic in the initial state: every q
variable at t=1 is created and, at first, FIXED to 0 (as if no state were possible yet), and
every initial-condition constraint's right-hand side is 0. `configure_compact_LP!` must be
called once (with the actual `x0`, `k0`) before the model is solved. Returns a
`CompactTemplate`, meant to be solved and reconfigured many times (see `make_policy_compact_lp`)
rather than rebuilt for every new state.
"""
struct CompactTemplate
    model::Model
    q::Dict{NTuple{6,Int},VariableRef}
    z::Dict{Tuple{Int,Int},VariableRef}
    m_start::Dict{Tuple{Int,Int,Int},VariableRef}
    free::Dict{Tuple{Int,Int,Int},VariableRef}
    ic::Dict{Tuple{Int,Int},ConstraintRef}    # (m, x) -> initial-condition constraint
    active_k::Vector{Int}                     # active_k[p]: the only k not fixed to 0 at t=1, -1 if none yet
end

function build_compact_template(fd::FluidData, t0_cyc::Int, H::Int)
    inst = fd.inst; M = fd.M; K = inst.n_poles; T = inst.T_period
    maxK = maximum(fd.K0)

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    set_optimizer_attribute(model, "Method", 1)      # dual simplex: supports warm start across re-solves

    # ── State-action variables (identical to fluid_lp_cg.build_master) ────────────
    q = Dict{NTuple{6,Int},VariableRef}()
    obj = AffExpr(0.0)
    for m in 1:M, t in 1:H, mode in 1:2, x in 1:fd.n_states[m]
        K0 = fd.K0[fd.pole_of[m]]
        for k in qrange(K0), κ in qrange(k)
            v = @variable(model, lower_bound = 0)
            q[(m, x, mode, k, κ, t)] = v
            t == 1 && fix(v, 0.0; force = true)   # every k fixed at first; configure_compact_LP! unfixes one
            add_to_expression!(obj, local_cost(fd, m, x, mode), v)
        end
    end

    # initial condition: right-hand side is 0 for now, set by configure_compact_LP!. Summing
    # over every k (not just k0[p]) keeps this constraint's structure independent of k0: only
    # the k0[p] slice is ever unfixed at solve time, so the sum is unaffected by the others
    # being fixed to 0, and k0 can change between re-solves without touching this constraint.
    ic = Dict{Tuple{Int,Int},ConstraintRef}()
    for m in 1:M, x in 1:fd.n_states[m]
        p = fd.pole_of[m]; K0 = fd.K0[p]
        ic[(m, x)] = @constraint(model,
            sum(q[(m, x, mode, k, κ, 1)] for mode in 1:2, k in qrange(K0), κ in qrange(k)) == 0)
    end

    # flow constraints on the state (x, k)
    for t in 2:H, m in 1:M
        p = fd.pole_of[m]; K0 = fd.K0[p]
        P = fd.P[m]
        reset = stage_cyc(t - 1, t0_cyc) == N_PERIODS_PER_YEAR
        for x in 1:fd.n_states[m], kn in qrange(K0)
            lhs = sum(q[(m, x, mode, kn, κ, t)] for mode in 1:2, κ in qrange(kn))
            if reset && kn != K0
                @constraint(model, lhs == 0)
                continue
            end
            rhs = AffExpr(0.0)
            for xp in 1:fd.n_states[m], mode in 1:2
                prob = mode == 1 ? P[x, xp] : (x == 1 ? 1.0 : 0.0)
                prob < 1e-12 && continue
                if reset
                    for k in qrange(K0), κ in qrange(k)
                        add_to_expression!(rhs, prob, q[(m, xp, mode, k, κ, t - 1)])
                    end
                else
                    for κ in 0:QUOTA_STEP:(K0 - kn)
                        add_to_expression!(rhs, prob, q[(m, xp, mode, kn + κ, κ, t - 1)])
                    end
                end
            end
            @constraint(model, lhs == rhs)
        end
    end

    # the copies of the quota inside a pole share the same (k, κ) distribution (Section 5.3)
    for p in 1:K, m in fd.comps_of_pole[p][2:end], t in 1:H
        m1 = fd.comps_of_pole[p][1]; K0 = fd.K0[p]
        for k in qrange(K0), κ in qrange(k)
            @constraint(model,
                sum(q[(m, x, mode, k, κ, t)] for x in 1:fd.n_states[m], mode in 1:2) ==
                sum(q[(m1, x, mode, k, κ, t)] for x in 1:fd.n_states[m1], mode in 1:2))
        end
    end

    # ── Relaxed schedule variables, one copy per period t (Eqs. (10)-(12), relaxed) ──
    last_start = [T - fd.d[m] + 1 - inst.margin for m in 1:M]
    first_start = FIRST_START_DAY   # earliest admissible start day (lead time to get parts, ...)

    z       = Dict{Tuple{Int,Int},VariableRef}()          # (m, t)
    m_start = Dict{Tuple{Int,Int,Int},VariableRef}()      # (m, τ, t)
    any_m   = Dict{Tuple{Int,Int,Int},VariableRef}()      # (p, τ, t)
    free    = Dict{Tuple{Int,Int,Int},VariableRef}()      # (p, τ, t)
    w       = Dict{Tuple{Int,Int,Int},VariableRef}()      # (p, τ, t)
    ksel    = Dict{Tuple{Int,Int,Int},VariableRef}()      # (p, κ, t)

    for t in 1:H
        for m in 1:M
            z[(m, t)] = @variable(model, lower_bound = 0, upper_bound = 1)
            if last_start[m] < first_start
                fix(z[(m, t)], 0.0; force = true)
                continue
            end
            for τ in first_start:last_start[m]
                m_start[(m, τ, t)] = @variable(model, lower_bound = 0, upper_bound = 1)
            end
            @constraint(model, sum(m_start[(m, τ, t)] for τ in first_start:last_start[m]) == z[(m, t)])
        end
        for p in 1:K, τ in 1:T
            any_m[(p, τ, t)] = @variable(model, lower_bound = 0, upper_bound = 1)
            free[(p, τ, t)]  = @variable(model, lower_bound = 0, upper_bound = 1)
            w[(p, τ, t)]     = @variable(model, lower_bound = 0, upper_bound = 1)
        end
        for p in 1:K, κ in qrange(maxK)
            ksel[(p, κ, t)] = @variable(model, lower_bound = 0, upper_bound = 1)
            κ > fd.K0[p] && fix(ksel[(p, κ, t)], 0.0; force = true)
        end
    end

    ong(m, τ, t) = sum((m_start[(m, s, t)] for s in max(first_start, τ - fd.d[m] + 1):min(τ, last_start[m])
                        if haskey(m_start, (m, s, t))); init = AffExpr(0.0))

    for t in 1:H
        for τ in 1:T
            @constraint(model, sum(fd.rho[m] * ong(m, τ, t) for m in 1:M) <= inst.R)   # (11)
            for p in 1:K
                for m in fd.comps_of_pole[p]
                    @constraint(model, any_m[(p, τ, t)] >= ong(m, τ, t))
                end
                @constraint(model, w[(p, τ, t)] <= free[(p, τ, t)])
                @constraint(model, w[(p, τ, t)] <= any_m[(p, τ, t)])
                @constraint(model, w[(p, τ, t)] >= free[(p, τ, t)] + any_m[(p, τ, t)] - 1)
            end
        end
        for p in 1:K   # (12)
            @constraint(model, sum(ksel[(p, κ, t)] for κ in qrange(maxK)) == 1)
            @constraint(model, sum(free[(p, τ, t)] for τ in 1:T) ==
                                sum(κ * ksel[(p, κ, t)] for κ in qrange(maxK)))
        end
    end

    # ── Coupling constraints (Appendix C.4) ────────────────────────────────────────
    for m in 1:M, t in 1:H
        K0 = fd.K0[fd.pole_of[m]]
        @constraint(model,
            sum(q[(m, x, 2, k, κ, t)] for x in 1:fd.n_states[m], k in qrange(K0), κ in qrange(k)) <= z[(m, t)])
    end
    for p in 1:K, t in 1:H
        m1 = fd.comps_of_pole[p][1]; K0 = fd.K0[p]
        @constraint(model,
            sum(κ * q[(m1, x, mode, k, κ, t)] for x in 1:fd.n_states[m1], mode in 1:2, k in qrange(K0), κ in qrange(k)) ==
            sum(free[(p, τ, t)] for τ in 1:T))
    end

    # ── Joint cost term of the objective (Appendix C.1) ─────────────────────────────
    for t in 1:H, p in 1:K, τ in 1:T
        pen = inst.poles[p].daily_penalty[τ]
        add_to_expression!(obj, pen, any_m[(p, τ, t)])
        add_to_expression!(obj, -pen, w[(p, τ, t)])
    end

    @objective(model, Min, obj)

    return CompactTemplate(model, q, z, m_start, free, ic, fill(-1, K))
end

"""
    configure_compact_LP!(fd, tmpl, x0, k0)

Points `tmpl` (built once by `build_compact_template`) at the state `(x0, k0)`: updates the
right-hand side of every initial-condition constraint to match `x0`, and re-fixes the t=1 q
variables so that only the `k0[p]` slice of each pole `p` is free (every other k is fixed to
0), unfixing/fixing just the two slices that change (`tmpl.active_k[p]` -> `k0[p]`) rather than
touching every k. Mutates `tmpl.model` in place: the previous solve's basis is kept, so the
next `optimize!` warm starts from it.
"""
function configure_compact_LP!(fd::FluidData, tmpl::CompactTemplate, x0::Vector{Int}, k0::Vector{Int})
    for m in 1:fd.M, x in 1:fd.n_states[m]
        set_normalized_rhs(tmpl.ic[(m, x)], x == x0[m] ? 1.0 : 0.0)
    end
    for p in 1:fd.inst.n_poles
        k0[p] == tmpl.active_k[p] && continue
        for m in fd.comps_of_pole[p], x in 1:fd.n_states[m], mode in 1:2
            if tmpl.active_k[p] != -1
                for κ in qrange(tmpl.active_k[p])
                    fix(tmpl.q[(m, x, mode, tmpl.active_k[p], κ, 1)], 0.0; force = true)
                end
            end
            for κ in qrange(k0[p])
                unfix(tmpl.q[(m, x, mode, k0[p], κ, 1)])
                set_lower_bound(tmpl.q[(m, x, mode, k0[p], κ, 1)], 0.0)
            end
        end
        tmpl.active_k[p] = k0[p]
    end
    return tmpl
end

"""
    build_compact_LP(fd, x0, k0, t0_cyc, H)

One-shot convenience wrapper (a fresh template, configured once): `build_compact_template` +
`configure_compact_LP!`. `make_policy_compact_lp` does not use this -- it keeps and reconfigures
one template per `(t0_cyc, H)` across every state, to warm start (see the module header).
"""
function build_compact_LP(fd::FluidData, x0::Vector{Int}, k0::Vector{Int}, t0_cyc::Int, H::Int)
    tmpl = build_compact_template(fd, t0_cyc, H)
    configure_compact_LP!(fd, tmpl, x0, k0)
    return tmpl.model, (q = tmpl.q, z = tmpl.z, m_start = tmpl.m_start, free = tmpl.free)
end

function solve_compact_LP!(model::Model)
    set_optimizer_attribute(model, "Method", 1)
    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
        set_optimizer_attribute(model, "Method", 2)
        optimize!(model)
        status = termination_status(model)
    end
    return status
end

# ─────────────────────────────────────────────────────────────────────────────
# Policy extraction: round the relaxed first-stage solution to a deterministic action
#
# The rounded solution of a relaxed LP is not necessarily an admissible joint action
# (Section 3.5): rounding the maintenance decision and the preferred start day of every
# component independently can violate the technician-capacity constraint (11), and can
# scatter maintenance windows over many different days, defeating the very purpose of the
# penalty-free quota, which "encourages maintenance activities to be grouped together in
# time" (Section 5.1). We therefore repair the rounded solution with a simple greedy
# heuristic, as prescribed in Section 3.5 ("we instead reconstruct an admissible action
# using a repair heuristic"):
#   - the maintenance decision is read from the ACTUAL replacement mass
#     Σ_{x,k,κ} q[m,x,2,k,κ,1] (mode 2 = maintained), not from z[m,1]: the coupling
#     constraint only bounds z FROM BELOW by this mass (Appendix C.4), so z is free to
#     float above it at no cost whenever the corresponding schedule can be covered by free
#     days, which happens routinely and would otherwise cause the repair heuristic to
#     "maintain" components that the LP never actually chose to replace;
#   - components with replacement mass > 0.5 are candidates for maintenance, examined in
#     decreasing order of technician requirement (the hardest to fit are placed first);
#   - each candidate FIRST tries to join a day already used by another component of the
#     same pole (ranked by its own relaxed weight on that day), and only opens a new day
#     (its own best day) if none of the already-open days fits (11); this concentrates
#     maintenance on as few distinct days as possible;
#   - if no day fits at all the component is simply not maintained this period (it will
#     be reconsidered at the next re-optimisation, as fluid policies are re-solved every
#     period);
#   - free days are then chosen, per pole, directly among the days that end up under
#     maintenance (rather than from the LP's free[τ] variable, which was optimised for
#     the fractional schedule and may no longer line up with the repaired one), capped at
#     the quota k[pole] actually remaining and prioritising the highest-penalty days.
# ─────────────────────────────────────────────────────────────────────────────

function extract_policy_compact(fd::FluidData, vars, k::Vector{Int}, T::Int)
    q = vars.q; m_start = vars.m_start
    inst = fd.inst

    replacement_mass(m) = sum(value(q[(m, x, 2, kk, κ, 1)])
                              for x in 1:fd.n_states[m]
                              for kk in qrange(fd.K0[fd.pole_of[m]]) for κ in qrange(kk))
    candidates = [m for m in 1:fd.M if replacement_mass(m) > 0.5]
    sort!(candidates; by = m -> -fd.rho[m])

    start   = zeros(Int, fd.M)
    demand  = zeros(T)
    opened  = [Int[] for _ in 1:inst.n_poles]   # days already used by this pole

    for m in candidates
        weights = Dict(τ => value(m_start[(m, τ, 1)]) for τ in 1:T if haskey(m_start, (m, τ, 1)))
        isempty(weights) && continue
        p = fd.pole_of[m]

        fits(τ) = τ + fd.d[m] - 1 <= T &&
                  all(demand[s] + fd.rho[m] <= inst.R + 1e-9 for s in τ:(τ + fd.d[m] - 1))

        chosen = 0
        for τ in sort(opened[p]; by = τ -> -get(weights, τ, -Inf))
            fits(τ) && (chosen = τ; break)
        end
        if chosen == 0
            for τ in sort(collect(keys(weights)); by = τ -> -weights[τ])
                fits(τ) && (chosen = τ; break)
            end
        end
        chosen == 0 && continue

        start[m] = chosen
        for s in chosen:(chosen + fd.d[m] - 1); demand[s] += fd.rho[m]; end
        chosen in opened[p] || push!(opened[p], chosen)
    end

    free_days = Vector{Vector{Int}}(undef, inst.n_poles)
    for p in 1:inst.n_poles
        days = Set{Int}()
        for m in fd.comps_of_pole[p]
            start[m] == 0 && continue
            for s in start[m]:(start[m] + fd.d[m] - 1); push!(days, s); end
        end
        days = collect(days)
        if length(days) > k[p]
            sort!(days; by = τ -> -inst.poles[p].daily_penalty[τ])
            days = days[1:k[p]]
        end
        free_days[p] = sort(days)
    end
    return Column(start, free_days)
end

# ─────────────────────────────────────────────────────────────────────────────
# Fluid policy with cache (same interface as make_policy_fluid_cg)
# ─────────────────────────────────────────────────────────────────────────────

"""
Returns a policy (x, k, t_global) -> (Column, solve_time), based on the Compact Fluid LP.
Same caching / logging interface as make_policy_fluid_cg, so it plugs directly into
simulation.jl and run_experiment.jl.
"""
function make_policy_compact_lp(fd::FluidData; H::Int = 6,
                                cache_file::Union{String,Nothing} = nothing,
                                log_solve::Function = info -> nothing,
                                heuristic_extraction::Bool = true)
    T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
    cache, next_cache_id = load_cache(cache_file)
    # One template per (t0_cyc, H_eff), reused and reconfigured (configure_compact_LP!) across
    # every state that shares it, instead of rebuilding a fresh model every time: this is what
    # lets Gurobi's simplex warm start from the previous solve's basis (module header).
    templates = Dict{Tuple{Int,Int},CompactTemplate}()

    function policy(x::Vector{Int}, k::Vector{Int}, t_global::Int)
        t0_cyc = (t_global - 1) % N_PERIODS_PER_YEAR + 1
        H_eff = min(H, T_total - t_global + 1)
        key = (Tuple(x), Tuple(k), t0_cyc, H_eff)
        if !haskey(cache, key)
            t_begin = time()
            tmpl = get!(() -> build_compact_template(fd, t0_cyc, H_eff), templates, (t0_cyc, H_eff))
            configure_compact_LP!(fd, tmpl, x, k)
            status = solve_compact_LP!(tmpl.model)
            status == MOI.OPTIMAL || error("compact LP not optimal: $status")
            lp_value = objective_value(tmpl.model)
            vars = (q = tmpl.q, z = tmpl.z, m_start = tmpl.m_start, free = tmpl.free)
            col = if heuristic_extraction
                q = vars.q
                mode_mass(m, mode) = sum(value(q[(m, x[m], mode, k[fd.pole_of[m]], κ, 1)])
                                          for κ in qrange(k[fd.pole_of[m]]))
                kappa_mass(p, κ) = sum(value(q[(fd.comps_of_pole[p][1], x[fd.comps_of_pole[p][1]],
                                                 mode, k[p], κ, 1)]) for mode in 1:2)
                extract_policy_heuristic(fd, x, k, mode_mass, kappa_mass)
            else
                extract_policy_compact(fd, vars, k, fd.inst.T_period)
            end
            elapsed = time() - t_begin

            cache[key] = (col.start, col.free_days, elapsed, lp_value)
            log_solve((key = key, time = elapsed, lp_value = lp_value,
                       iterations = 1, n_columns = 0, converged = true))
            append_cache_entry!(cache_file, next_cache_id, key, cache[key])
            return col, elapsed
        end
        start, free_days = cache[key][1], cache[key][2]
        return Column(start, free_days), 0.0
    end
    return policy
end
