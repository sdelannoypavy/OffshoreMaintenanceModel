# fluid_lp_cg.jl
#
# Full fluid LP (equation (4) of the paper) solved by column generation (Section 3.2,
# Appendix C), and the corresponding fluid policy (Section 3.5).
#
# Model (no weather scenario: the substations are always accessible):
#   - components m = (pole, component) are indexed by a flat index 1:M,
#   - local state of m: (x_m, k) where k is the remaining free-day quota of the pole of m
#     (one copy of k per component, see Section 5.3),
#   - local action of m: (mode, κ) where mode = 1 (not maintained) or 2 (maintained) and
#     κ = number of penalty-free days declared on the pole of m during the period,
#   - joint action b = "column": a start day for each maintained component and a set of free
#     days for each pole. It satisfies the technician capacity constraint on every day.
#   - joint cost f(b) = Σ_poles Σ_days penalty_τ (am_τ − w_τ).
#
# Design of the LP
#   min  Σ c_m(x,a) q^m_{x,a,t} + Σ_{b,t} f(b) μ_{b,t}
#   s.t. flow constraints (4b), initial condition (4d),
#        consistency (4c):  Σ_{x,k} q^m_{x,(k,κ),mode,t} = Σ_{b : A_m(b) = (mode,κ)} μ_{b,t},
#        equal (k,κ) distributions for the copies of the quota inside a pole (Section 5.3).

using JuMP, Gurobi, JLD2, Printf

@isdefined(GRB_ENV) || (GRB_ENV = Gurobi.Env())

# ─────────────────────────────────────────────────────────────────────────────
# Flat description of an instance
# ─────────────────────────────────────────────────────────────────────────────

struct FluidData
    inst::Instance
    M::Int                              # total number of components
    pole_of::Vector{Int}                # pole of each component
    comp_of::Vector{Int}                # index of the component inside its pole
    comps_of_pole::Vector{Vector{Int}}  # flat indices of the components of each pole
    n_states::Vector{Int}
    d::Vector{Int}                      # repair durations
    rho::Vector{Float64}                # technician requirement (share of the workforce)
    cc::Vector{Float64}                 # replacement cost
    K0::Vector{Int}                     # annual quota of free days of each pole
    δ::Vector{Vector{Float64}}          # δ[m][x]: expected failure penalty over a period
    P::Vector{Matrix{Float64}}          # period transition matrix of each component (to, from), cleaned
end

# ─── Policy cache in a csv file (cache_file ending with .csv) ───────────────────────────────────
# One row per decision: state, quota, t0_cyc, horizon, start, free_days, time, lp_value. The state,
# the quota and the start days are written "a-b-c"; the penalty-free days of the poles are separated
# by "/" and the days of a pole by spaces. A row is appended as soon as the decision is computed, and
# the file is read back at restart. Only the action (start days and free days) is used on a cache hit.
const POLICY_CSV_HEADER = "state,quota,t0_cyc,horizon,start,free_days,time,lp_value"

join_ints(v) = join(v, "-")
parse_ints(s) = isempty(s) ? Int[] : parse.(Int, split(s, "-"))

function load_cache_csv(cache_file::String)
    cache = Dict{Any,Any}()
    if isfile(cache_file)
        for l in readlines(cache_file)[2:end]
            r = split(l, ",")
            length(r) == 8 || continue   # line cut by an interruption
            key = (Tuple(parse_ints(r[1])), Tuple(parse_ints(r[2])), parse(Int, r[3]), parse(Int, r[4]))
            free = [isempty(p) ? Int[] : parse.(Int, split(p, " ")) for p in split(r[6], "/")]
            cache[key] = (parse_ints(r[5]), free, parse(Float64, r[7]), parse(Float64, r[8]))
        end
        @printf("  cache loaded: %d policies from %s\n", length(cache), cache_file)
    end
    return cache, Ref(0)
end

function append_cache_entry_csv!(cache_file::String, key, value)
    x, k, t0, H = key
    row = join([join_ints(collect(x)), join_ints(collect(k)), string(t0), string(H), join_ints(value[1]),
                join([join(p, " ") for p in value[2]], "/"), @sprintf("%.4f", value[3]), @sprintf("%.8g", value[4])], ",")
    for attempt in 1:60
        try
            (!isfile(cache_file) || filesize(cache_file) == 0) && open(io -> println(io, POLICY_CSV_HEADER), cache_file, "w")
            open(io -> println(io, row), cache_file, "a")
            return
        catch e
            (e isa SystemError && attempt < 60) || rethrow()
            sleep(1.0)
        end
    end
end

"""
    load_cache(cache_file) -> (cache::Dict, next_id::Ref{Int})

Loads a policy cache from `cache_file`. Entries are stored one per JLD2 dataset (names "e1",
"e2", ...) instead of as a single dataset holding the whole dictionary, so that later appending
a new entry (`append_cache_entry!`) only writes that one small dataset, without touching what is
already on disk. Files written by the old, single-dataset format (a "cache" key holding the
whole Dict) are still merged in transparently, so no existing cache needs migrating. `next_id`
is the highest entry number already used in `cache_file`, so that later calls to
`append_cache_entry!` do not collide with names already on disk when resuming a run.
"""
function load_cache(cache_file::Union{String,Nothing})
    !isnothing(cache_file) && endswith(cache_file, ".csv") && return load_cache_csv(cache_file)
    cache = Dict{Any,Any}()
    max_id = 0
    if !isnothing(cache_file) && isfile(cache_file)
        jldopen(cache_file, "r") do f
            for k in keys(f)
                if k == "cache"
                    merge!(cache, f["cache"])
                else
                    m = match(r"^e(\d+)$", k)
                    isnothing(m) && continue
                    entry_key, entry_value = f[k]
                    cache[entry_key] = entry_value
                    max_id = max(max_id, parse(Int, m[1]))
                end
            end
        end
        @printf("  cache loaded: %d policies from %s\n", length(cache), cache_file)
    end
    return cache, Ref(max_id)
end

"""
    append_cache_entry!(cache_file, next_id, key, value)

Appends one new `(key, value)` cache entry to `cache_file` as its own JLD2 dataset, without
re-serializing the entries already on disk -- unlike overwriting a single dataset holding the
whole cache dictionary, whose cost grows with the cache size (invisible on a small instance like
`one_pole_simple`, whose cache plateaus around a few thousand entries, but on a multi-pole
instance the cache can reach tens of thousands of entries, and re-saving it all then costs far
more than the LP solve that triggered it -- a compact-LP solve is tens of milliseconds,
rewriting a multi-megabyte JLD2 file is not). `next_id` is the counter returned by
`load_cache`.
"""
function append_cache_entry!(cache_file::Union{String,Nothing}, next_id::Ref{Int}, key, value)
    isnothing(cache_file) && return
    endswith(cache_file, ".csv") && return append_cache_entry_csv!(cache_file, key, value)
    next_id[] += 1
    # On Windows, opening the file fails while another process reads it (the simulations of the other
    # chunks read this cache at start, sim_range.jl): retry for up to about a minute.
    for attempt in 1:60
        try
            jldopen(cache_file, "a+") do f
                f["e$(next_id[])"] = (key, value)
            end
            return
        catch e
            (e isa SystemError && attempt < 60) || rethrow()
            sleep(1.0)
        end
    end
end

"Remove the negligible transition probabilities (numerical stability of the LP); the mass removed stays on the diagonal."
function clean_transitions(P::Matrix{Float64}; tol::Float64 = 1e-8)
    P = copy(P)
    for x in 1:size(P, 2)
        for xp in 1:size(P, 1)
            if xp != x && P[xp, x] < tol
                P[x, x] += P[xp, x]; P[xp, x] = 0.0
            end
        end
    end
    return P
end

"""
Expected penalty of component c of `pole` over one period, if it is not maintained and starts
the period in state x. The other components are assumed at full capacity (the cost of a pole
is approximated by the sum of the costs of its components).
"""
function failure_cost(pole::Pole, c::Int, x::Int, T::Int)
    v = zeros(pole.n[c]); v[x] = 1.0
    cost = 0.0
    for τ in 1:T
        loss = [(100.0 - pole.P_evac[c, s]) / 100 for s in 1:pole.n[c]]
        cost += pole.daily_penalty[τ] * sum(v .* loss)
        v = pole.P_daily[c] * v
    end
    return cost
end

"""
Expected penalty of component c of `pole` over one period, if it is not maintained and
stays in state x for the WHOLE period (no within-period degradation, `inst.simplified`):
the daily capacity-loss penalty of state x, summed over the T days of the period, with no
day-by-day evolution (contrast with `failure_cost`, which propagates `v` through
`pole.P_daily` every day). Matches simulate_period's simplified branch (simulation.jl)
exactly, so the additive fluid LP's own cost model has no gap with the true simulated cost
on a `simplified` instance.
"""
function failure_cost_simplified(pole::Pole, c::Int, x::Int, T::Int)
    loss = (100.0 - pole.P_evac[c, x]) / 100
    return loss * sum(pole.daily_penalty[τ] for τ in 1:T)
end

"""
    build_fluid_data(inst; alpha=1.0)

`alpha` scales δ (the per-component expected/period failure penalty, whichever of
`failure_cost`/`failure_cost_simplified` applies), i.e. δ_m(x) = alpha * (cost of leaving
m unmaintained this period in state x). `alpha = 1` (default) is the cost as defined
elsewhere in this file; `alpha` lets a caller reweight the failure penalty relative to the
replacement cost `cc_m` and the joint schedule cost `f(b)` (both left unscaled), e.g. to
study how sensitive a method's policy is to that trade-off. It is a parameter of this
FluidData, shared by every method built on top of it (fluid_cg, fluid_cg_light, compact,
fluid_cg_order2, lift_project all only ever read `fd.δ`, already scaled): to compare
methods at different alpha, build one FluidData per alpha and pass each to the methods
separately.
"""
function build_fluid_data(inst::Instance; alpha::Float64 = 1.0)
    pole_of = Int[]; comp_of = Int[]
    comps_of_pole = [Int[] for _ in 1:inst.n_poles]
    for k in 1:inst.n_poles, c in eachindex(inst.poles[k].n)
        push!(pole_of, k); push!(comp_of, c)
        push!(comps_of_pole[k], length(pole_of))
    end
    M = length(pole_of)
    pole(m) = inst.poles[pole_of[m]]
    return FluidData(inst, M, pole_of, comp_of, comps_of_pole,
        [pole(m).n[comp_of[m]] for m in 1:M],
        [pole(m).d[comp_of[m]] for m in 1:M],
        [pole(m).technician_share[comp_of[m]] for m in 1:M],
        [pole(m).replacement_cost[comp_of[m]] for m in 1:M],
        [p.free_quota for p in inst.poles],
        [[(c -> c < 1e-6 ? 0.0 : c)(alpha * (inst.simplified ? failure_cost_simplified : failure_cost)(pole(m), comp_of[m], x, inst.T_period))
          for x in 1:pole(m).n[comp_of[m]]]
         for m in 1:M],
        [clean_transitions(pole(m).P_period[comp_of[m]]) for m in 1:M])
end

"""
Local cost c_m(x, mode) of component m in state x (mode 1: not maintained, 2: maintained).
Original instances: δ_m(x) if not maintained, cc_m if maintained. Simplified instances: the
failure cost δ_m(x) is paid whatever the mode (it is the cost of the period spent in state x,
also when the component is maintained during the period), plus cc_m if maintained.
"""
local_cost(fd::FluidData, m::Int, x::Int, mode::Int) =
    mode == 1 ? fd.δ[m][x] : (fd.inst.simplified ? fd.δ[m][x] : 0.0) + fd.cc[m]

# ─────────────────────────────────────────────────────────────────────────────
# Columns (joint actions b)
# ─────────────────────────────────────────────────────────────────────────────

struct Column
    start::Vector{Int}                # start[m] = start day of component m, 0 if not maintained
    free_days::Vector{Vector{Int}}    # free_days[k] = sorted penalty-free days of pole k
end

Base.:(==)(a::Column, b::Column) = a.start == b.start && a.free_days == b.free_days
Base.hash(c::Column, h::UInt) = hash((c.start, c.free_days), h)

column_mode(col::Column, m::Int) = col.start[m] > 0 ? 2 : 1
column_kappa(col::Column, k::Int) = length(col.free_days[k])

empty_column(fd::FluidData) = Column(zeros(Int, fd.M), [Int[] for _ in 1:fd.inst.n_poles])

function singleton_column(fd::FluidData, m::Int, start_day::Int)
    col = empty_column(fd)
    col.start[m] = start_day
    return col
end

"Common mobilization cost of a joint action: MOBILIZATION_COST if at least one component is maintained (instances.jl)."
mobilization_cost(col::Column) = any(>(0), col.start) ? MOBILIZATION_COST : 0.0

"Joint cost f(b): penalty of the poles during the days of maintenance that are not free, plus the mobilization cost."
function column_penalty(fd::FluidData, col::Column)
    inst = fd.inst; total = mobilization_cost(col)
    for k in 1:inst.n_poles
        free = falses(inst.T_period)
        for τ in col.free_days[k]; free[τ] = true; end
        for τ in 1:inst.T_period
            active = any(col.start[m] > 0 && col.start[m] <= τ <= col.start[m] + fd.d[m] - 1
                         for m in fd.comps_of_pole[k])
            active && !free[τ] && (total += inst.poles[k].daily_penalty[τ])
        end
    end
    return total
end

"Capacity loss ℓ_m(n) ∈ [0,1] of component m in degradation state n (paper, Section 2.2)."
capacity_loss(fd::FluidData, m::Int, n::Int) =
    (100.0 - fd.inst.poles[fd.pole_of[m]].P_evac[fd.comp_of[m], n]) / 100

"""
    true_period_cost(fd, x, col) -> (penalty, replacement)

True cost Tc of a period of the paper (Section 2.2, Eq. (eq:intro-cost)), for the
degradation states `x` at the start of the period and the joint action `col`. There is no
degradation during the period: the state of component m on day τ is ñ_{m,τ} = 1 if its
repair is complete on day τ (start_m ≠ 0 and τ ≥ start_m + d_m), and x[m] otherwise. The
capacity loss of pole s on day τ is Cl^s_τ = m^s_τ + (1 − m^s_τ) max_{m ∈ s} ℓ_m(ñ_{m,τ}),
where m^s_τ = 1 iff some component of s is under repair on day τ, and the penalty of s is
the sum of π_τ Cl^s_τ over the days τ that are NOT declared penalty-free (the declared days
exempt every capacity loss, due to maintenance or to degradation). `penalty` is the sum of
the penalties of the poles and `replacement` the sum of the replacement costs of the
components repaired, plus the mobilization cost (mobilization_cost), so that Tc = penalty + replacement.
"""
function true_period_cost(fd::FluidData, x::Vector{Int}, col::Column)
    inst = fd.inst; T = inst.T_period
    penalty = 0.0
    for p in 1:inst.n_poles
        free = falses(T)
        for τ in col.free_days[p]; free[τ] = true; end
        for τ in 1:T
            free[τ] && continue
            ms = fd.comps_of_pole[p]
            under_repair = any(col.start[m] > 0 && col.start[m] <= τ <= col.start[m] + fd.d[m] - 1 for m in ms)
            if under_repair
                cl = 1.0
            else
                cl = maximum(capacity_loss(fd, m, (col.start[m] > 0 && τ >= col.start[m] + fd.d[m]) ? 1 : x[m])
                             for m in ms)
            end
            penalty += inst.poles[p].daily_penalty[τ] * cl
        end
    end
    replacement = sum(fd.cc[m] for m in 1:fd.M if col.start[m] > 0; init = 0.0) + mobilization_cost(col)
    return penalty, replacement
end

"""
The LP's own (additive, "sum of subsystem costs plus joint action cost") approximation of
the period cost, evaluated at the ACTUALLY REALISED pre-period state `x` and the chosen
column `col` -- as opposed to the true cost computed by `simulate_period`/`pole_penalty`
of simulation.jl, which is non-additive (a pole's capacity is the min over its components,
and a component under maintenance never contributes an ADDITIONAL failure penalty on top,
since the pole is already fully down that day). This is exactly the model gap noted in
Appendix C.4 of the paper ("In the numerical experiments section we assumed an
approximated problem where the cost is the sum of subsystem costs and the cost of the
joint action. In the reality the cost is global."): the two costs can differ whenever a
maintained component and a simultaneously-degraded, non-maintained component of the same
pole overlap on the same day, since `δ_m` of the non-maintained one is charged in full by
the LP regardless of what else is happening to its pole that period.

Matches, term for term, the objective of build_master (Section 2: `Σ c_m(x,a) q` for the
state-action cost, `column_penalty` for the joint schedule cost `f(b)`) and, if `Δ_pairs`
is given (order-2, Section 4), also its pairwise correction, restricted to the pairs whose
BOTH components are not maintained by `col` (the only mode combination Δ_pairs was
computed for) and evaluated at their actually realised states.
"""
function lp_approx_cost(fd::FluidData, x::Vector{Int}, col::Column;
                        Δ_pairs::Union{Dict{Tuple{Int,Int},Matrix{Float64}},Nothing} = nothing)
    c = column_penalty(fd, col)
    for m in 1:fd.M
        c += local_cost(fd, m, x[m], column_mode(col, m))
    end
    if !isnothing(Δ_pairs)
        for ((m1, m2), Δ) in Δ_pairs
            (column_mode(col, m1) == 1 && column_mode(col, m2) == 1) || continue
            c += Δ[x[m1], x[m2]]
        end
    end
    return c
end

# ─────────────────────────────────────────────────────────────────────────────
# Restricted master LP
# ─────────────────────────────────────────────────────────────────────────────

# The quota resets at the start of every year: stage t+1 is the first of a year iff
# stage t is the last (6th) period of a year.
stage_cyc(t::Int, t0_cyc::Int) = (t0_cyc - 1 + t - 1) % N_PERIODS_PER_YEAR + 1

mutable struct MasterLP
    model::Model
    H::Int
    cons::Dict{NTuple{4,Int},ConstraintRef}   # consistency constraints (m, mode, κ, t), summed over k
    q::Dict{NTuple{6,Int},VariableRef}        # (m, x, mode, k, κ, t)
    μ::Vector{Vector{VariableRef}}            # μ[col_id][t]
    cols::Vector{Column}                      # columns already added, in μ's col_id order
    ic::Dict{Tuple{Int,Int},ConstraintRef}    # initial-condition constraint (m, x) -> constraint
    active_k::Vector{Int}                     # active_k[p]: the only k not fixed to 0 at t=1, -1 if none yet
end

"""
    build_master(fd, t0_cyc, H)

Builds the master LP for a given `(t0_cyc, H)`, generic in the initial state (x0, k0): every
q variable at t=1 is created and, at first, FIXED to 0, and every initial-condition
constraint's right-hand side is 0. `configure_master!` must be called once (with the actual
x0, k0) before the model is solved. The point of separating the two is that the SAME model can
then be reused and reconfigured for every state sharing this `(t0_cyc, H)`, instead of being
rebuilt from scratch (see `configure_master!` and `make_policy_fluid_cg`), the same way
`compact_lp.jl`'s `build_compact_template`/`configure_compact_LP!` let the Compact Fluid LP
warm-start across states.
"""
function build_master(fd::FluidData, t0_cyc::Int, H::Int)
    M = fd.M
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    # Dual simplex, not the barrier: column generation needs duals tied to an actual basic
    # (vertex) solution. The barrier without crossover converges to an interior point of a
    # highly degenerate optimal face here, whose "duals" are not associated with any
    # specific basis and can vary between re-solves, letting the master spread μ_{b,t} mass
    # over several near-tied columns instead of a single vertex — this is what caused the
    # mismatched-column artifact tracked down earlier (a near-tie broken inconsistently by
    # argmax). Dual simplex always returns a genuine vertex, with well-defined duals, and
    # warm-starts cheaply across column-generation iterations (only a few columns/variables
    # are added between successive solves, so the previous basis stays nearly optimal) AND,
    # via configure_master!, across different states sharing this (t0_cyc, H) template.
    set_optimizer_attribute(model, "Method", 1)

    q = Dict{NTuple{6,Int},VariableRef}()
    obj = AffExpr(0.0)
    for m in 1:M, t in 1:H, mode in 1:2, x in 1:fd.n_states[m]
        K0 = fd.K0[fd.pole_of[m]]
        for k in qrange(K0), κ in qrange(k)
            v = @variable(model, lower_bound = 0)
            q[(m, x, mode, k, κ, t)] = v
            t == 1 && fix(v, 0.0; force = true)   # every k fixed at first; configure_master! unfixes one
            add_to_expression!(obj, local_cost(fd, m, x, mode), v)
        end
    end
    @objective(model, Min, obj)

    # initial condition: right-hand side is 0 for now, set by configure_master!. Summing over
    # every k (not just k0[p]) keeps this constraint's structure independent of k0: only the
    # k0[p] slice is ever unfixed at solve time, so the sum is unaffected by the others being
    # fixed to 0, and k0 can change between re-solves without touching this constraint.
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
                prob < 1e-12 && continue   # negligible probabilities (numerical stability)
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

    # Consistency with the joint action (paper, Eq. (30c)): for every local action (mode, κ),
    # summed over the state (x, k), equal to the mass of the columns b with A_m(b) = mode and
    # κ_p(b) = κ (see add_column!). The quota k does not appear on the right-hand side.
    cons = Dict{NTuple{4,Int},ConstraintRef}()
    for t in 1:H, m in 1:M, mode in 1:2
        K0 = fd.K0[fd.pole_of[m]]
        for κ in qrange(K0)
            cons[(m, mode, κ, t)] = @constraint(model,
                sum(q[(m, x, mode, k, κ, t)] for x in 1:fd.n_states[m], k in κ:QUOTA_STEP:K0) == 0)
        end
    end

    # The copies of the quota inside a pole share the same (k, κ) distribution (paper,
    # Eq. (30d)), exactly as in compact_lp.jl.
    for p in 1:fd.inst.n_poles, m in fd.comps_of_pole[p][2:end], t in 1:H
        m1 = fd.comps_of_pole[p][1]; K0 = fd.K0[p]
        for k in qrange(K0), κ in qrange(k)
            @constraint(model,
                sum(q[(m, x, mode, k, κ, t)] for x in 1:fd.n_states[m], mode in 1:2) ==
                sum(q[(m1, x, mode, k, κ, t)] for x in 1:fd.n_states[m1], mode in 1:2))
        end
    end

    return MasterLP(model, H, cons, q, Vector{VariableRef}[], Column[], ic, fill(-1, fd.inst.n_poles))
end

"""
    configure_master!(fd, master, x0, k0)

Points `master` (built once by `build_master`) at the state `(x0, k0)`: updates the
right-hand side of every initial-condition constraint to match `x0`, and re-fixes the t=1 q
variables so that only the `k0[p]` slice of each pole `p` is free (every other k fixed to 0),
unfixing/fixing just the two slices that change (`master.active_k[p]` -> `k0[p]`) rather than
touching every k. Mutates `master.model` in place: the previous solve's basis (and every
column already added) is kept, so the next `solve_master!` warm-starts from it, the same way
`compact_lp.jl`'s `configure_compact_LP!` warm-starts the Compact Fluid LP.
"""
function configure_master!(fd::FluidData, master::MasterLP, x0::Vector{Int}, k0::Vector{Int})
    for m in 1:fd.M, x in 1:fd.n_states[m]
        set_normalized_rhs(master.ic[(m, x)], x == x0[m] ? 1.0 : 0.0)
    end
    for p in 1:fd.inst.n_poles
        k0[p] == master.active_k[p] && continue
        for m in fd.comps_of_pole[p], x in 1:fd.n_states[m], mode in 1:2
            if master.active_k[p] != -1
                for κ in qrange(master.active_k[p])
                    fix(master.q[(m, x, mode, master.active_k[p], κ, 1)], 0.0; force = true)
                end
            end
            for κ in qrange(k0[p])
                unfix(master.q[(m, x, mode, k0[p], κ, 1)])
                set_lower_bound(master.q[(m, x, mode, k0[p], κ, 1)], 0.0)
            end
        end
        master.active_k[p] = k0[p]
    end
    return master
end

"""
Add the variables μ_{b,t}, t = 1..H, of column b to the master LP: μ_{b,t} enters
cons[m, A_m(b), κ_p(b), t] with coefficient −1 for every component m of every pole p
(paper, Eq. (30c)).
"""
function add_column!(fd::FluidData, master::MasterLP, col::Column)
    f = column_penalty(fd, col)
    vars = Vector{VariableRef}(undef, master.H)
    for t in 1:master.H
        v = @variable(master.model, lower_bound = 0)   # μ_{b,t}
        set_objective_coefficient(master.model, v, f)
        for p in 1:fd.inst.n_poles
            κ = column_kappa(col, p)
            for m in fd.comps_of_pole[p]
                set_normalized_coefficient(master.cons[(m, column_mode(col, m), κ, t)], v, -1.0)
            end
        end
        vars[t] = v
    end
    push!(master.μ, vars)
end

"""
Solve the master LP and check that an optimal dual solution is actually available before
returning, per the JuMP column-generation tutorial
(https://jump.dev/JuMP.jl/stable/tutorials/algorithms/cutting_stock_column_generation/),
which checks `is_solved_and_feasible(model; dual = true)` after every master solve, before
ever calling `dual.(...)` — the LP being solved and feasible is not, by itself, a
guarantee that duals were computed (e.g. if the optimizer's dual solve fails while the
primal succeeds). Dual simplex is used by default (a genuine vertex solution, with duals
tied to that specific basis, see build_master); on failure, fall back on the barrier
method with crossover enabled (Method 2, default Crossover), which is slower but also
returns a vertex solution rather than an interior point.
"""
function solve_master!(master::MasterLP)
    set_optimizer_attribute(master.model, "Method", 1)
    optimize!(master.model)
    if !is_solved_and_feasible(master.model; dual = true)
        set_optimizer_attribute(master.model, "Method", 2)
        optimize!(master.model)
        assert_is_solved_and_feasible(master.model; dual = true)
    end
end

"Duals λ[m, mode, κ+1, t] of the consistency constraints."
function consistency_duals(fd::FluidData, master::MasterLP)
    maxK = maximum(fd.K0)
    λ = zeros(fd.M, 2, maxK + 1, master.H)
    for ((m, mode, κ, t), con) in master.cons
        λ[m, mode, κ + 1, t] = dual(con)
    end
    return λ
end

# ─────────────────────────────────────────────────────────────────────────────
# Pricing problem (13): min_b  f(b) + Σ_m λ_t^{m, A_m(b), κ_{pole(m)}(b)}
#
# The consistency constraints of the master are indexed by (m, mode, κ) only (paper,
# Eq. (30c)), so pricing chooses, per pole, the number κ of declared free days, through the
# selection variable ksel[pole, κ] and the McCormick linearisation y[m, κ] = z_m ∧ ksel[pole(m), κ].
# ─────────────────────────────────────────────────────────────────────────────

function solve_pricing(fd::FluidData, λ_t::Array{Float64,3};
                       time_limit::Float64 = 60.0, n_pool::Int = 1, pool_tol::Float64 = 1e-6)
    inst = fd.inst; M = fd.M; T = inst.T_period; K = inst.n_poles

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    set_time_limit_sec(model, time_limit)
    # n_pool > 1: Gurobi keeps up to n_pool solutions, and every other solution of negative
    # reduced cost (below -pool_tol) is returned as an extra column (third return value).
    n_pool > 1 && set_optimizer_attribute(model, "PoolSolutions", n_pool)

    last_start = [T - fd.d[m] + 1 - inst.margin for m in 1:M]   # constraint (10)
    first_start = FIRST_START_DAY   # earliest admissible start day (lead time to get parts, ...)
    # Start days are restricted to the grid first_start, first_start+QUOTA_STEP, ... Since
    # every repair duration d_m is a multiple of QUOTA_STEP, a schedule using only these start
    # days never has worse cost than an unrestricted one here: with the current (uniform) daily
    # penalty, cost depends only on how many capacity-loss days are penalised, not on which
    # days, so any feasible schedule can be shifted onto the grid without changing its cost,
    # and without breaking technician-capacity feasibility (shifting starts to grid points only
    # tightens/aligns overlaps, it cannot create a new one). See the red remark in
    # implemented_formulations.tex for the caveat if the daily penalty stops being uniform.
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

    # component m is under maintenance on day τ iff it started during the last d_m days
    ong(m, τ) = sum((ms[(m, s)] for s in max(first_start, τ - fd.d[m] + 1):min(τ, last_start[m])
                     if haskey(ms, (m, s))); init = AffExpr(0.0))

    @variable(model, 0 <= any_m[1:K, 1:T] <= 1)
    @variable(model, free[1:K, 1:T], Bin)
    @variable(model, 0 <= w[1:K, 1:T] <= 1)

    # ksel[p, κ] = 1 iff κ free days are declared on pole p
    ksel = Dict{Tuple{Int,Int},VariableRef}()
    for p in 1:K, κ in qrange(fd.K0[p])
        ksel[(p, κ)] = @variable(model, binary = true)
    end

    # ong(m, ·) is piecewise-constant with breakpoints only at the start-day grid (every start
    # is a grid point and every duration is a multiple of QUOTA_STEP, so every end-of-maintenance
    # day is also a grid point); checking the technician-capacity constraint on the grid alone is
    # therefore exactly equivalent to checking it on every day of the period. No component can
    # be under maintenance before first_start, so days 1..first_start-1 need no check either.
    for τ in first_start:QUOTA_STEP:T
        @constraint(model, sum(fd.rho[m] * ong(m, τ) for m in 1:M) <= inst.R)   # (11)
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
    for p in 1:K   # (12)
        @constraint(model, sum(ksel[(p, κ)] for κ in qrange(fd.K0[p])) == 1)
        @constraint(model, sum(free[p, τ] for τ in 1:T) == sum(κ * ksel[(p, κ)] for κ in qrange(fd.K0[p])))
    end

    obj = AffExpr(0.0)
    for p in 1:K, τ in 1:T
        pen = inst.poles[p].daily_penalty[τ]
        add_to_expression!(obj, pen, any_m[p, τ])
        add_to_expression!(obj, -pen, w[p, τ])
    end
    y = Dict{Tuple{Int,Int},VariableRef}()   # y[m, κ] = z_m ∧ ksel[pole(m), κ]
    for m in 1:M
        p = fd.pole_of[m]
        for κ in qrange(fd.K0[p])
            y[(m, κ)] = @variable(model, lower_bound = 0, upper_bound = 1)
            @constraint(model, y[(m, κ)] <= z[m])
            @constraint(model, y[(m, κ)] <= ksel[(p, κ)])
            @constraint(model, y[(m, κ)] >= z[m] + ksel[(p, κ)] - 1)
            λ1 = λ_t[m, 1, κ + 1]; λ2 = λ_t[m, 2, κ + 1]
            add_to_expression!(obj, λ1, ksel[(p, κ)])
            add_to_expression!(obj, λ2 - λ1, y[(m, κ)])
        end
    end
    @objective(model, Min, obj)
    optimize!(model)

    status = termination_status(model)
    if !(status == MOI.OPTIMAL || (status == MOI.TIME_LIMIT && has_values(model)))
        return nothing, Inf, Column[]
    end

    start = zeros(Int, M)
    for m in 1:M
        value(z[m]) > 0.5 || continue
        for τ in starts_grid(m)
            value(ms[(m, τ)]) > 0.5 && (start[m] = τ)
        end
    end
    free_days = [[τ for τ in 1:T if value(free[p, τ]) > 0.5] for p in 1:K]
    extras = Column[]
    for r in 2:result_count(model)
        objective_value(model; result = r) < -pool_tol || continue
        st = zeros(Int, M)
        for m in 1:M
            value(z[m]; result = r) > 0.5 || continue
            for τ in starts_grid(m)
                value(ms[(m, τ)]; result = r) > 0.5 && (st[m] = τ)
            end
        end
        push!(extras, Column(st, [[τ for τ in 1:T if value(free[p, τ]; result = r) > 0.5] for p in 1:K]))
    end
    return Column(start, free_days), objective_value(model), extras
end

# ─────────────────────────────────────────────────────────────────────────────
# Column generation (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

"""
Solve the full fluid LP by column generation for the initial state (x0, k0), over `H` stages,
on an already-built and `configure_master!`-ed `master`. Returns the first-stage joint action
with the largest weight, and statistics. `master.cols` (and its μ variables) persist
across calls when `master` itself is reused across states (see `make_policy_fluid_cg`): only
columns genuinely new to this `master` are ever added.
"""
function solve_fluid_cg(fd::FluidData, master::MasterLP, x0::Vector{Int}, k0::Vector{Int}, H::Int;
                        max_iter::Int = 50, tol::Float64 = 1e-6,
                        time_limit_pricing::Float64 = 60.0,
                        heuristic_extraction::Bool = true)
    t_begin = time()
    begin
        function push_col!(col)
            col in master.cols && return
            push!(master.cols, col); add_column!(fd, master, col)
        end
        if isempty(master.cols)
            push_col!(empty_column(fd))
            for m in 1:fd.M; push_col!(singleton_column(fd, m, 1)); end
        end

        iterations = 0
        converged = false   # true iff the loop stopped because no column has a reduced cost < -tol
        for iter in 1:max_iter
            iterations = iter
            solve_master!(master)
            λ = consistency_duals(fd, master)
            n_new = 0
            for t in 1:H
                col, rc = solve_pricing(fd, λ[:, :, :, t]; time_limit = time_limit_pricing)
                (isnothing(col) || rc >= -tol || col in master.cols) && continue
                push_col!(col); n_new += 1
            end
            if n_new == 0
                converged = true
                break
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
    end
    elapsed = time() - t_begin
    return best, (time = elapsed, lp_value = lp_value, iterations = iterations, converged = converged,
                  n_columns = length(master.cols), cols = master.cols)
end

# ─────────────────────────────────────────────────────────────────────────────
# Fluid policy with cache
# ─────────────────────────────────────────────────────────────────────────────

"""
Returns a policy (x, k, t_global) -> Column, where x is the flat vector of the degradation
states and k the vector of remaining quotas of the poles. Each new LP solved is timed and
reported through `log_solve(info)`. Solutions are cached (in memory and in `cache_file`).

When `warm_start` is true (default), one `MasterLP` template is built per `(t0_cyc, H_eff)`
and reused/reconfigured (`configure_master!`) across every state sharing it, instead of being
rebuilt from scratch: this lets Gurobi's dual simplex warm-start from the previous solve's
basis, and lets column generation reuse every column already found for that `(t0_cyc, H_eff)`
across states (not just within one state's own iterations), the same way `compact_lp.jl`
warm-starts the Compact Fluid LP (see its module header). `warm_start = false` rebuilds a
fresh, column-less master for every state instead.
"""
function make_policy_fluid_cg(fd::FluidData; H::Int = 6, warm_start::Bool = true,
                              cache_file::Union{String,Nothing} = nothing,
                              log_solve::Function = info -> nothing,
                              max_iter::Int = 50, time_limit_pricing::Float64 = 60.0,
                              heuristic_extraction::Bool = true)
    T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
    # key -> (start, free_days, solve_time, lp_value, iterations, n_columns, converged)
    cache, next_cache_id = load_cache(cache_file)
    masters = Dict{Tuple{Int,Int},MasterLP}()   # (t0_cyc, H_eff) -> template, reused iff warm_start

    function policy(x::Vector{Int}, k::Vector{Int}, t_global::Int)
        t0_cyc = (t_global - 1) % N_PERIODS_PER_YEAR + 1
        H_eff = min(H, T_total - t_global + 1)      # the fluid horizon is truncated at the end
        key = (Tuple(x), Tuple(k), t0_cyc, H_eff)
        if !haskey(cache, key)
            master = warm_start ? get!(() -> build_master(fd, t0_cyc, H_eff), masters, (t0_cyc, H_eff)) :
                                   build_master(fd, t0_cyc, H_eff)
            configure_master!(fd, master, x, k)
            col, info = solve_fluid_cg(fd, master, x, k, H_eff; max_iter = max_iter,
                                       time_limit_pricing = time_limit_pricing,
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
