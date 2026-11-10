# lift_project.jl
#
# Lift-and-project cuts (Section 4.3), added by a cutting-plane loop on top of the Full
# Fluid LP solved by column generation (fluid_lp_cg.jl). See Section 5 ("Lift-and-project
# cuts") of implemented_formulations.tex for the full derivation; this file implements
# cut (C') exactly as stated there.
#
# Proposition 10 of the paper bounds q^{m1}_{x1,a,t} + q^{m2}_{x2,a,t} by mu_{a,t} plus a
# correction term obtained by propagating a min of order-1 marginals from stage t-1
# through the transition kernel -- well suited to a D-MDP where every subsystem shares the
# exact same global action a. Our combinatorial joint action b assigns independent local
# modes A_m1(b) != A_m2(b) in general, so there is no shared "a" to use directly. Fix a
# pole p, two distinct components m1 != m2 of p (only components of the SAME pole share a
# common quota, so cross-pole pairs give no such cut), modes a1, a2 in {1,2}, a valid
# quota pair 0 <= kappa <= k <= K0(p), and a stage t >= 2. With
#   E := {X_{m1,t}=x1, A_{m1}(B_t)=a1, K_{p,t}=k, Kappa_{p,t}=kappa}
#   F := {X_{m2,t}=x2, A_{m2}(B_t)=a2, K_{p,t}=k, Kappa_{p,t}=kappa}
# so that P[E] = q_{m1,x1,a1,k,kappa,t} and P[F] = q_{m2,x2,a2,k,kappa,t} exactly, we bound
# each side of P[E]+P[F] = P[E union F] + P[E inter F] with the tool best suited to it:
#
#   * E union F implies {(A_m1(B_t)=a1 or A_m2(B_t)=a2), K_{p,t}=k, Kappa_{p,t}=kappa}.
#     Since the master LP does not index mu_{b,t} by the quota k (paper, Eq. (30)), we bound
#     its probability by the smaller of two linear expressions (both valid):
#       sigma^b := sum_{b in B': A_m1(b)=a1 or A_m2(b)=a2, kappa_p(b)=kappa} mu_{b,t}
#                  (dropping the condition K_{p,t}=k),
#       sigma^q := sum_{x,a} q_{m1,x,a,k,kappa,t} = P[K_{p,t}=k, Kappa_{p,t}=kappa]
#                  (dropping the condition on the actions),
#     sigma := min(sigma^b, sigma^q), the minimum being chosen at the current LP solution when
#     the cut is generated, and the corresponding linear expression added to the LP;
#
#   * E inter F implies {X_{m1,t}=x1, X_{m2,t}=x2, K_{p,t}=k} (dropping the action/kappa
#     conditions this time), whose probability is bounded, exactly as in the proof of
#     Proposition 10, by propagating a min of order-1 marginals of stage t-1 through the
#     transition kernel:
#       tau^+_{m1,m2,t}(x1,x2;k)
#         := sum_{(k',kappa') -> k} sum_{a1',a2' in {1,2}} sum_{x1',x2'}
#              p_{m1}^{a1'}(x1|x1') p_{m2}^{a2'}(x2|x2')
#              min(q_{m1,x1',a1',k',kappa',t-1}, q_{m2,x2',a2',k',kappa',t-1})
#     where "(k',kappa') -> k" are the same predecessor quota states summed by the Flow /
#     Flow-reset constraints of fluid_lp_cg.jl for target k at stage t.
#
# giving cut (C'):
#   q_{m1,x1,a1,k,kappa,t} + q_{m2,x2,a2,k,kappa,t}
#     <= sigma_{m1,m2,(a1,a2,k,kappa),t} + tau^+_{m1,m2,t}(x1,x2;k)
# for every occupation measure realisable by an admissible policy.
#
# (An earlier version of this file bounded P[E inter F] by the EXACT probability that a
# single column maintains m1 AND m2 simultaneously in the prescribed way. Under a shared
# technician-capacity constraint this is needlessly hard to satisfy -- few, if any,
# generated columns maintain two specific components in the exact same way on the exact
# same day -- making that bound very loose. sigma/tau^+ above replace it with the tighter,
# better-suited pairing described in implemented_formulations.tex.)
#
# Cutting-plane algorithm: solve the column-generation LP to optimality, enumerate every
# candidate cut and evaluate its violation at the current solution, add the few most
# violated ones (as genuine linear constraints -- see add_cut!), and repeat until none is
# violated (or an iteration budget is reached).
#
# Requires fluid_lp_cg.jl to be loaded first (provides FluidData, Column, MasterLP,
# build_master, add_column!, solve_master!, consistency_duals, solve_pricing, stage_cyc,
# empty_column, singleton_column, column_mode, column_kappa, GRB_ENV, ...).

using JuMP, Printf

# ─────────────────────────────────────────────────────────────────────────────
# Transitions and predecessor quota states
# ─────────────────────────────────────────────────────────────────────────────

"""
Transition matrix (to, from) of component `m` under mode `a`: `fd.P[m]` if `a == 1`
(not maintained), or the deterministic reset to state 1 if `a == 2` (maintained), matching
p^2_m(x|x') = 1{x=1} of fluid_lp_cg.jl.
"""
function mode_transition(fd::FluidData, m::Int, a::Int)
    a == 1 && return fd.P[m]
    n = fd.n_states[m]
    P = zeros(n, n)
    P[1, :] .= 1.0
    return P
end

"(k', κ') pairs at stage t-1 from which pole p's quota can become exactly k at stage t."
function predecessors_kκ(fd::FluidData, p::Int, t0_cyc::Int, t::Int, k::Int)
    K0 = fd.K0[p]
    if stage_cyc(t - 1, t0_cyc) == N_PERIODS_PER_YEAR   # year reset between t-1 and t
        k != K0 && return Tuple{Int,Int}[]
        return [(kp, κp) for kp in qrange(K0) for κp in qrange(kp)]
    else
        return [(k + κp, κp) for κp in 0:QUOTA_STEP:(K0 - k)]
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Evaluating candidate cuts at the current LP solution
# ─────────────────────────────────────────────────────────────────────────────

"Snapshot value(q[...]) once for the whole model (a single pass), reused by every candidate cut."
q_value_snapshot(master::MasterLP) = Dict(key => value(v) for (key, v) in master.q)

"Snapshot value.(μ[col_id]) once for the whole model, reused by every candidate cut."
μ_snapshot(master::MasterLP) = [value.(vec) for vec in master.μ]

qv(qval, m, x, a, k, κ, t) = get(qval, (m, x, a, k, κ, t), 0.0)

"""
σ^b_{m1,m2,(a1,a2,κ),t} = Σ_{b: A_{m1}(b)=a1 or A_{m2}(b)=a2, κ_p(b)=κ} μ_{b,t}, evaluated at the
snapshot `μval` (see the header of this file).
"""
function σb_value(μval, cols::Vector{Column}, p::Int, m1::Int, m2::Int,
                  a1::Int, a2::Int, κ::Int, t::Int)
    s = 0.0
    for (col_id, col) in enumerate(cols)
        column_kappa(col, p) == κ || continue
        (column_mode(col, m1) == a1 || column_mode(col, m2) == a2) || continue
        s += μval[col_id][t]
    end
    return s
end

"σ^q_{m1,(k,κ),t} = Σ_{x,a} q_{m1,x,a,k,κ,t} = P[K_{p,t}=k, Kappa_{p,t}=κ], at the snapshot `qval`."
σq_value(qval, fd::FluidData, m1::Int, k::Int, κ::Int, t::Int) =
    sum(qv(qval, m1, x, a, k, κ, t) for x in 1:fd.n_states[m1], a in 1:2)

"""
τ^+ matrix (n_{m1} x n_{m2}) for the pair (m1, m2) of the same pole `p`, at target quota
`k` and stage `t`, computed from the snapshot `qval` of the previous stage (independent of
κ, a1, a2: those are summed/fixed elsewhere, see implemented_formulations.tex). Uses
matrix products P1 * min(v1, v2') * P2' rather than a quadruple loop over (x1,x2,x1',x2').
"""
function τ_plus_matrix(qval, fd::FluidData, p::Int, m1::Int, m2::Int, t0_cyc::Int, t::Int, k::Int)
    n1 = fd.n_states[m1]; n2 = fd.n_states[m2]
    τ = zeros(n1, n2)
    for (kp, κp) in predecessors_kκ(fd, p, t0_cyc, t, k)
        for a1 in 1:2, a2 in 1:2
            v1 = [qv(qval, m1, x, a1, kp, κp, t - 1) for x in 1:n1]
            all(iszero, v1) && continue
            v2 = [qv(qval, m2, x, a2, kp, κp, t - 1) for x in 1:n2]
            all(iszero, v2) && continue
            MinMat = min.(v1, v2')                       # n1 x n2, MinMat[x1',x2'] = min(v1[x1'],v2[x2'])
            P1 = mode_transition(fd, m1, a1)              # n1 x n1 (to, from)
            P2 = mode_transition(fd, m2, a2)              # n2 x n2 (to, from)
            τ .+= P1 * MinMat * P2'
        end
    end
    return τ
end

struct CutCandidate
    p::Int; m1::Int; m2::Int; x1::Int; x2::Int; a1::Int; a2::Int; k::Int; κ::Int; t::Int
    use_σb::Bool          # true: σ = σ^b (columns), false: σ = σ^q (quota marginal)
    violation::Float64
end

"""
Enumerate every candidate cut (C') (same-pole pairs only, t >= 2) and return those
violated by more than `tol`, sorted by decreasing violation.
"""
function find_violated_cuts(fd::FluidData, master::MasterLP, cols::Vector{Column},
                            t0_cyc::Int, H::Int; tol::Float64 = 1e-6)
    qval = q_value_snapshot(master)
    μval = μ_snapshot(master)
    candidates = CutCandidate[]
    for p in 1:fd.inst.n_poles
        comps = fd.comps_of_pole[p]
        K0 = fd.K0[p]
        for i in eachindex(comps), j in (i + 1):length(comps)
            m1, m2 = comps[i], comps[j]
            for t in 2:H, k in qrange(K0)
                τ = τ_plus_matrix(qval, fd, p, m1, m2, t0_cyc, t, k)   # independent of κ, a1, a2
                for κ in qrange(k), a1 in 1:2, a2 in 1:2
                    σb = σb_value(μval, cols, p, m1, m2, a1, a2, κ, t)
                    σq = σq_value(qval, fd, m1, k, κ, t)
                    use_σb = σb <= σq
                    σ = min(σb, σq)
                    for x1 in 1:fd.n_states[m1], x2 in 1:fd.n_states[m2]
                        lhs = qv(qval, m1, x1, a1, k, κ, t) + qv(qval, m2, x2, a2, k, κ, t)
                        viol = lhs - (σ + τ[x1, x2])
                        viol > tol && push!(candidates, CutCandidate(p, m1, m2, x1, x2, a1, a2, k, κ, t, use_σb, viol))
                    end
                end
            end
        end
    end
    sort!(candidates; by = c -> -c.violation)
    return candidates
end

# ─────────────────────────────────────────────────────────────────────────────
# Adding a cut to the master LP
# ─────────────────────────────────────────────────────────────────────────────

"""
Add cut `c` to `master.model` as a genuine linear constraint: σ is a plain linear
expression in the already-existing μ (σ^b) or q (σ^q) variables, whichever was chosen when
the cut was generated, while
each min(q1, q2) term of τ^+ is realised by a fresh variable z >= 0 with z <= q1 and
z <= q2 -- since nothing else constrains z and a larger z only relaxes the cut, the solver
pushes every z to exactly min(q1, q2) at the optimum, making the linear constraint below
equivalent to (C').
"""
function add_cut!(master::MasterLP, fd::FluidData, cols::Vector{Column}, c::CutCandidate, t0_cyc::Int)
    model = master.model
    lhs = master.q[(c.m1, c.x1, c.a1, c.k, c.κ, c.t)] + master.q[(c.m2, c.x2, c.a2, c.k, c.κ, c.t)]

    rhs = AffExpr(0.0)   # σ: σ^b (in the μ variables) or σ^q (in the q variables)
    if c.use_σb
        for (col_id, col) in enumerate(cols)
            column_kappa(col, c.p) == c.κ || continue
            (column_mode(col, c.m1) == c.a1 || column_mode(col, c.m2) == c.a2) || continue
            add_to_expression!(rhs, master.μ[col_id][c.t])
        end
    else
        for x in 1:fd.n_states[c.m1], a in 1:2
            add_to_expression!(rhs, master.q[(c.m1, x, a, c.k, c.κ, c.t)])
        end
    end

    for (kp, κp) in predecessors_kκ(fd, c.p, t0_cyc, c.t, c.k)   # τ^+: min-linearised, propagated from t-1
        for a1p in 1:2, a2p in 1:2
            P1 = mode_transition(fd, c.m1, a1p)
            P2 = mode_transition(fd, c.m2, a2p)
            for x1p in 1:fd.n_states[c.m1], x2p in 1:fd.n_states[c.m2]
                coeff = P1[c.x1, x1p] * P2[c.x2, x2p]
                coeff < 1e-12 && continue
                z = @variable(model, lower_bound = 0)
                @constraint(model, z <= master.q[(c.m1, x1p, a1p, kp, κp, c.t - 1)])
                @constraint(model, z <= master.q[(c.m2, x2p, a2p, kp, κp, c.t - 1)])
                add_to_expression!(rhs, coeff, z)
            end
        end
    end
    @constraint(model, lhs <= rhs)
end

# ─────────────────────────────────────────────────────────────────────────────
# Column generation + lift-and-project cutting-plane loop
# ─────────────────────────────────────────────────────────────────────────────

"""
Solve the Full Fluid LP by column generation, then add the most violated lift-and-project
cuts (C'), re-solving by column generation after each round, until none is violated (or
`max_cut_iter` rounds are used), on an already-built and `configure_master!`-ed `master`.
Same interface (return value, keyword names) as solve_fluid_cg, plus `n_cuts` /
`cut_iterations` in the info tuple. Cuts added to `master.model` are genuinely valid linear
inequalities regardless of the current state (Proposition 10-style argument): when `master`
is reused across states (see `make_policy_lift_project`), cuts found for one state remain in
the model (and valid) for every later state sharing its `(t0_cyc, H)`, on top of the usual
column/basis reuse -- at the cost of the model growing with every new cut added over the run.
"""
function solve_fluid_cg_liftproject(fd::FluidData, master::MasterLP, x0::Vector{Int}, k0::Vector{Int},
                                    t0_cyc::Int, H::Int;
                                    max_iter::Int = 50, tol::Float64 = 1e-6,
                                    time_limit_pricing::Float64 = 60.0,
                                    max_cut_iter::Int = 10, cut_tol::Float64 = 1e-6,
                                    max_cuts_per_iter::Int = 3,
                                    heuristic_extraction::Bool = true)
    t_begin = time()
    function push_col!(col)
        col in master.cols && return
        push!(master.cols, col); add_column!(fd, master, col)
    end
    if isempty(master.cols)
        push_col!(empty_column(fd))
        for m in 1:fd.M; push_col!(singleton_column(fd, m, 1)); end
    end

    total_cg_iterations = 0
    cg_converged = false
    function run_column_generation!()
        for iter in 1:max_iter
            total_cg_iterations += 1
            solve_master!(master)
            λ = consistency_duals(fd, master)
            n_new = 0
            for t in 1:H
                col, rc = solve_pricing(fd, λ[:, :, :, t]; time_limit = time_limit_pricing)
                (isnothing(col) || rc >= -tol || col in master.cols) && continue
                push_col!(col); n_new += 1
            end
            if n_new == 0
                cg_converged = true
                return
            end
        end
        cg_converged = false
        solve_master!(master)   # columns may have been added on the very last iteration
    end

    n_cuts = 0
    cut_iterations = 0
    for cut_iter in 1:max_cut_iter
        cut_iterations = cut_iter
        run_column_generation!()
        candidates = find_violated_cuts(fd, master, master.cols, t0_cyc, H; tol = cut_tol)
        isempty(candidates) && break
        for c in candidates[1:min(max_cuts_per_iter, length(candidates))]
            add_cut!(master, fd, master.cols, c, t0_cyc)
            n_cuts += 1
        end
    end
    solve_master!(master)

    weights = [value(master.μ[i][1]) for i in eachindex(master.cols)]
    best = master.cols[argmax(weights)]
    if heuristic_extraction
        mode_mass(m, mode) = sum(value(master.q[(m, x0[m], mode, k0[fd.pole_of[m]], κ, 1)])
                                  for κ in qrange(k0[fd.pole_of[m]]))
        kappa_mass(p, κ) = sum(value(master.q[(fd.comps_of_pole[p][1], x0[fd.comps_of_pole[p][1]],
                                                 mode, k0[p], κ, 1)]) for mode in 1:2)
        best = extract_policy_heuristic(fd, x0, k0, mode_mass, kappa_mass)
    end
    lp_value = objective_value(master.model)
    elapsed = time() - t_begin
    return best, (time = elapsed, lp_value = lp_value, iterations = total_cg_iterations,
                  converged = cg_converged, n_columns = length(master.cols), cols = master.cols,
                  n_cuts = n_cuts, cut_iterations = cut_iterations)
end

# ─────────────────────────────────────────────────────────────────────────────
# Fluid policy with cache (same interface as make_policy_fluid_cg)
# ─────────────────────────────────────────────────────────────────────────────

"""
Returns a policy (x, k, t_global) -> (Column, solve_time), based on the Full Fluid LP
strengthened by lift-and-project cuts (Section 4.3) and solved by column generation. Same
caching / logging interface as make_policy_fluid_cg.

When `warm_start` is true (default), one `MasterLP` template is built per `(t0_cyc, H_eff)`
and reused/reconfigured (`configure_master!`) across every state sharing it -- carrying over
its basis, its columns AND every lift-and-project cut added so far -- instead of being
rebuilt from scratch for every state (see `solve_fluid_cg_liftproject`'s docstring for the
trade-off this introduces). `warm_start = false` rebuilds a fresh, column- and cut-less master
for every state instead.
"""
function make_policy_lift_project(fd::FluidData; H::Int = 6, warm_start::Bool = true,
                                  cache_file::Union{String,Nothing} = nothing,
                                  log_solve::Function = info -> nothing,
                                  max_iter::Int = 50, time_limit_pricing::Float64 = 60.0,
                                  max_cut_iter::Int = 10, max_cuts_per_iter::Int = 3,
                                  heuristic_extraction::Bool = true)
    T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
    cache, next_cache_id = load_cache(cache_file)
    masters = Dict{Tuple{Int,Int},MasterLP}()   # (t0_cyc, H_eff) -> template, reused iff warm_start

    function policy(x::Vector{Int}, k::Vector{Int}, t_global::Int)
        t0_cyc = (t_global - 1) % N_PERIODS_PER_YEAR + 1
        H_eff = min(H, T_total - t_global + 1)
        key = (Tuple(x), Tuple(k), t0_cyc, H_eff)
        if !haskey(cache, key)
            master = warm_start ? get!(() -> build_master(fd, t0_cyc, H_eff), masters, (t0_cyc, H_eff)) :
                                   build_master(fd, t0_cyc, H_eff)
            configure_master!(fd, master, x, k)
            col, info = solve_fluid_cg_liftproject(fd, master, x, k, t0_cyc, H_eff; max_iter = max_iter,
                                                   time_limit_pricing = time_limit_pricing,
                                                   max_cut_iter = max_cut_iter,
                                                   max_cuts_per_iter = max_cuts_per_iter,
                                                   heuristic_extraction = heuristic_extraction)
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
