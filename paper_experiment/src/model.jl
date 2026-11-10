# model.jl
#
# Common definitions of the MDP of paper.tex, used by every method:
#   - FluidData: flat description of an instance, the components of all the substations being
#     indexed by m ∈ 1:M,
#   - Column: a joint action b (start day of every repair, declared days of every substation),
#   - the costs of a period (true cost of the full model, cost of the simplified model) and the
#     separable approximation c̃_m of Section "Joint fluid LP",
#   - the transitions of the local states (degradation and remaining quota),
#   - the policy wrapper shared by the fluid policies (cache of the decisions already computed).
#
# Notation of the code: a substation is a "pole", the maintenance decision of a component is its
# "mode" (1 = not maintained, 2 = maintained), and κ is the number of declared penalty-free days.

using JuMP, Gurobi, Printf

@isdefined(GRB_ENV) || (GRB_ENV = Gurobi.Env())

# ─────────────────────────────────────────────────────────────────────────────
# Flat description of an instance
# ─────────────────────────────────────────────────────────────────────────────

struct FluidData
    inst::Instance
    M::Int                              # total number of components
    pole_of::Vector{Int}                # substation of each component
    comp_of::Vector{Int}                # index of the component inside its substation
    comps_of_pole::Vector{Vector{Int}}  # components of each substation
    n_states::Vector{Int}               # N_m
    d::Vector{Int}                      # repair duration d_m
    rho::Vector{Float64}                # technician share ρ_m
    cc::Vector{Float64}                 # replacement cost c^rep_m
    K0::Vector{Int}                     # annual quota of each substation
    P::Vector{Matrix{Float64}}          # transition matrix over one period of each component, P[to, from]
end

function build_fluid_data(inst::Instance)
    pole_of = Int[]; comp_of = Int[]; comps_of_pole = [Int[] for _ in 1:inst.n_poles]
    for p in 1:inst.n_poles, c in eachindex(inst.poles[p].n)
        push!(pole_of, p); push!(comp_of, c); push!(comps_of_pole[p], length(pole_of))
    end
    M = length(pole_of)
    pole(m) = inst.poles[pole_of[m]]
    return FluidData(inst, M, pole_of, comp_of, comps_of_pole,
                     [pole(m).n[comp_of[m]] for m in 1:M],
                     [pole(m).d[comp_of[m]] for m in 1:M],
                     [pole(m).technician_share[comp_of[m]] for m in 1:M],
                     [pole(m).replacement_cost[comp_of[m]] for m in 1:M],
                     [p.free_quota for p in inst.poles],
                     [pole(m).P_period[comp_of[m]] for m in 1:M])
end

"Capacity loss ℓ_m(n) ∈ [0, 1] of component m in degradation state n."
capacity_loss(fd::FluidData, m::Int, n::Int) =
    (100.0 - fd.inst.poles[fd.pole_of[m]].P_evac[fd.comp_of[m], n]) / 100

"Daily penalty π of the instance. The separable approximation of the cost assumes that it is constant."
function constant_daily_penalty(fd::FluidData)
    pens = unique(vcat([p.daily_penalty for p in fd.inst.poles]...))
    length(pens) == 1 || error("the cost approximation assumes a constant daily penalty")
    return pens[1]
end

"Failure cost of component m in state n over a period without declared days, T π ℓ_m(n) (eq:intro-failure-cost with κ = 0)."
failure_cost(fd::FluidData, m::Int, n::Int) = fd.inst.T_period * constant_daily_penalty(fd) * capacity_loss(fd, m, n)

# ─────────────────────────────────────────────────────────────────────────────
# Joint actions
# ─────────────────────────────────────────────────────────────────────────────

struct Column
    start::Vector{Int}                # start[m] = start day of the repair of m, 0 if m is not maintained
    free_days::Vector{Vector{Int}}    # free_days[p] = declared penalty-free days of substation p
end

Base.:(==)(a::Column, b::Column) = a.start == b.start && a.free_days == b.free_days
Base.hash(c::Column, h::UInt) = hash((c.start, c.free_days), h)

"Maintenance decision of component m: 1 if it is not maintained, 2 if it is maintained."
column_mode(col::Column, m::Int) = col.start[m] > 0 ? 2 : 1
"Number κ of declared days of substation p."
column_kappa(col::Column, p::Int) = length(col.free_days[p])

empty_column(fd::FluidData) = Column(zeros(Int, fd.M), [Int[] for _ in 1:fd.inst.n_poles])

function singleton_column(fd::FluidData, m::Int, start_day::Int)
    col = empty_column(fd)
    col.start[m] = start_day
    return col
end

"Grid of the start days of component m: FIRST_START_DAY, FIRST_START_DAY + STEP, … as long as the repair ends in the period."
start_days(fd::FluidData, m::Int) = collect(FIRST_START_DAY:STEP:(fd.inst.T_period - fd.d[m] + 1))

"True iff component m of the schedule `start` is under repair on day τ."
under_repair(fd::FluidData, start::Vector{Int}, m::Int, τ::Int) = start[m] > 0 && start[m] <= τ <= start[m] + fd.d[m] - 1

# ─────────────────────────────────────────────────────────────────────────────
# Costs of a period
# ─────────────────────────────────────────────────────────────────────────────

"""
Joint action cost f(b) = c^maint (eq:intro-maintenance-cost): penalty of the maintenance days that
are not declared penalty-free, summed over the substations.
"""
function joint_action_cost(fd::FluidData, col::Column)
    total = 0.0
    for p in 1:fd.inst.n_poles, τ in 1:fd.inst.T_period
        τ in col.free_days[p] && continue
        any(under_repair(fd, col.start, m, τ) for m in fd.comps_of_pole[p]) && (total += fd.inst.poles[p].daily_penalty[τ])
    end
    return total
end

"""
True cost of a period of the full model (eq:intro-cost), for the degradation states `x` at the
start of the period and the joint action `col`. The state of component m on day τ is 1 if its repair
is complete and x[m] otherwise. The capacity loss of a substation is 1 on its maintenance days and
the maximum of the capacity losses of its components otherwise. It is penalised on the days that
are not declared. Returns (penalty, replacement cost).
"""
function full_model_period_cost(fd::FluidData, x::Vector{Int}, col::Column)
    penalty = 0.0
    for p in 1:fd.inst.n_poles, τ in 1:fd.inst.T_period
        τ in col.free_days[p] && continue
        ms = fd.comps_of_pole[p]
        loss = any(under_repair(fd, col.start, m, τ) for m in ms) ? 1.0 :
               maximum(capacity_loss(fd, m, (col.start[m] > 0 && τ >= col.start[m] + fd.d[m]) ? 1 : x[m]) for m in ms)
        penalty += fd.inst.poles[p].daily_penalty[τ] * loss
    end
    replacement = sum(fd.cc[m] for m in 1:fd.M if col.start[m] > 0; init = 0.0)
    return penalty, replacement
end

"""
Cost of a period of the simplified model: the failure costs of all the components, maintained or
not, plus the maintenance cost (number of maintenance days of every substation). Returns
(penalty, replacement cost).
"""
function simplified_model_period_cost(fd::FluidData, x::Vector{Int}, col::Column)
    penalty = sum(failure_cost(fd, m, x[m]) for m in 1:fd.M) + joint_action_cost(fd, col)
    replacement = sum(fd.cc[m] for m in 1:fd.M if col.start[m] > 0; init = 0.0)
    return penalty, replacement
end

"Cost of a period, (penalty, replacement cost), of the model of the instance."
period_cost(fd::FluidData, x::Vector{Int}, col::Column) =
    fd.inst.simplified ? simplified_model_period_cost(fd, x, col) : full_model_period_cost(fd, x, col)

"""
Separable approximation c̃_m(x, a) of the cost (eq:cost_approx), for the local state x = (n, k) and
the local action a = (mode, κ): δ c^rep_m + (T − κ) π ℓ_m(n). On the simplified model (κ = 0), it is
the exact local cost of the component.
"""
local_cost(fd::FluidData, m::Int, n::Int, mode::Int, κ::Int) =
    (mode == 2 ? fd.cc[m] : 0.0) + (fd.inst.T_period - κ) * constant_daily_penalty(fd) * capacity_loss(fd, m, n)

# ─────────────────────────────────────────────────────────────────────────────
# Transitions of the local states
# ─────────────────────────────────────────────────────────────────────────────

# The transition probabilities below TRANSITION_TOL are moved onto the diagonal in the LPs (the
# simulator keeps the exact transitions). Without this, the order-2 LP, whose transitions of the pairs
# are products of two probabilities, has coefficients down to 1e-16 and Gurobi fails to solve it.
TRANSITION_TOL = 1e-8
const ORDER2_TRANSITION_TOL = 1e-5   # tolerance used by the order-2 LP (jflp_cg.jl)
const TRANSITION_CACHE = Dict{Tuple{UInt,Int,Float64},Matrix{Float64}}()

"Remove the transition probabilities below `tol`, the mass removed staying on the diagonal."
function clean_transitions(P::Matrix{Float64}; tol::Float64)
    P = copy(P)
    for x in 1:size(P, 2), xp in 1:size(P, 1)
        if xp != x && P[xp, x] < tol
            P[x, x] += P[xp, x]; P[xp, x] = 0.0
        end
    end
    return P
end

"Probability p_m(n | n', mode) that component m goes from state n' to state n over a period (a repair resets it to 1)."
function trans_prob(fd::FluidData, m::Int, n::Int, np::Int, mode::Int)
    mode == 2 && return n == 1 ? 1.0 : 0.0
    P = get!(() -> clean_transitions(fd.P[m]; tol = TRANSITION_TOL), TRANSITION_CACHE, (objectid(fd), m, TRANSITION_TOL))
    return P[n, np]
end

"Index of the period of the year (1..6) of stage t of a fluid horizon starting at the period t0_cyc of the year."
stage_cyc(t::Int, t0_cyc::Int) = (t0_cyc - 1 + t - 1) % N_PERIODS_PER_YEAR + 1

"Quota dynamics φ_t(k, κ) (eq:intro-quota-dynamics) at stage t: reset to K0 at the start of a new year."
quota_next(t::Int, t0_cyc::Int, k::Int, κ::Int, K0::Int) = stage_cyc(t, t0_cyc) == N_PERIODS_PER_YEAR ? K0 : k - κ

"Pairs (k', κ') at stage t − 1 from which the quota becomes k at stage t (flow constraints)."
function quota_predecessors(t::Int, t0_cyc::Int, k::Int, K0::Int)
    if stage_cyc(t - 1, t0_cyc) == N_PERIODS_PER_YEAR
        return k == K0 ? [(kp, κp) for kp in qrange(K0) for κp in qrange(kp)] : Tuple{Int,Int}[]
    end
    return [(k + κp, κp) for κp in 0:STEP:(K0 - k)]
end

# ─────────────────────────────────────────────────────────────────────────────
# Fluid policies (Section "Fluid policies")
# ─────────────────────────────────────────────────────────────────────────────

# A policy is a function (x, k, t) -> (Column, computing time), where x is the vector of the
# degradation states, k the vector of the remaining quotas and t ∈ 1:120 the period. The decision
# only depends on (x, k), on the period of the year t0_cyc and on the remaining horizon H_eff, so that
# it is computed once and reused (paragraph "Policy evaluation").

"Key of a decision in the cache."
decision_key(x, k, t0_cyc, H_eff) = (Tuple(x), Tuple(k), t0_cyc, H_eff)

"""
    make_fluid_policy(fd, solve; H, cache_file, log_solve)

Policy of a fluid LP of horizon `H`. `solve(x, k, t0_cyc, H_eff)` returns the applied joint action and
a named tuple with at least the fields `time`, `lp_value`, `t_milp` (pricing MILPs) and `t_lp` (LPs).
The decisions are kept in memory and appended to `cache_file` (csv, see `load_cache`), and every
new decision is passed to `log_solve`.
"""
function make_fluid_policy(fd::FluidData, solve::Function; H::Int = 6,
                           cache_file::Union{String,Nothing} = nothing, log_solve::Function = info -> nothing)
    T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
    cache = load_cache(cache_file)
    function policy(x::Vector{Int}, k::Vector{Int}, t::Int)
        t0_cyc = (t - 1) % N_PERIODS_PER_YEAR + 1
        H_eff = min(H, T_total - t + 1)
        key = decision_key(x, k, t0_cyc, H_eff)
        haskey(cache, key) && return Column(cache[key][1], cache[key][2]), 0.0
        col, info = solve(x, k, t0_cyc, H_eff)
        cache[key] = (col.start, col.free_days, info.time, info.lp_value)
        append_cache_entry!(cache_file, key, cache[key])
        log_solve((key = key, time = info.time, lp_value = info.lp_value, t_milp = info.t_milp, t_lp = info.t_lp))
        return col, info.time
    end
    return policy
end

# Cache of the decisions in a csv file, one row per decision:
#   state,quota,t0_cyc,horizon,start,free_days,time,lp_value
# The state, the quota and the start days are written "a-b-c", the declared days of the
# substations are separated by "/" and the days of a substation by spaces.
const CACHE_HEADER = "state,quota,t0_cyc,horizon,start,free_days,time,lp_value"

join_ints(v) = join(v, "-")
parse_ints(s) = isempty(s) ? Int[] : parse.(Int, split(s, "-"))

"Decisions stored in `cache_file` (empty if the file does not exist or is `nothing`)."
function load_cache(cache_file::Union{String,Nothing})
    cache = Dict{Any,Any}()
    (isnothing(cache_file) || !isfile(cache_file)) && return cache
    for l in readlines(cache_file)[2:end]
        r = split(l, ",")
        length(r) == 8 || continue   # line cut by an interruption
        key = decision_key(parse_ints(r[1]), parse_ints(r[2]), parse(Int, r[3]), parse(Int, r[4]))
        free = [isempty(p) ? Int[] : parse.(Int, split(p, " ")) for p in split(r[6], "/")]
        cache[key] = (parse_ints(r[5]), free, parse(Float64, r[7]), parse(Float64, r[8]))
    end
    @printf("  cache loaded: %d decisions from %s\n", length(cache), cache_file)
    return cache
end

"Append one decision to `cache_file` (retried for about a minute if another process is reading the file)."
function append_cache_entry!(cache_file::Union{String,Nothing}, key, value)
    isnothing(cache_file) && return
    x, k, t0, H = key
    row = join([join_ints(collect(x)), join_ints(collect(k)), string(t0), string(H), join_ints(value[1]),
                join([join(p, " ") for p in value[2]], "/"), @sprintf("%.4f", value[3]), @sprintf("%.8g", value[4])], ",")
    for attempt in 1:60
        try
            (!isfile(cache_file) || filesize(cache_file) == 0) && open(io -> println(io, CACHE_HEADER), cache_file, "w")
            open(io -> println(io, row), cache_file, "a")
            return
        catch e
            (e isa SystemError && attempt < 60) || rethrow()
            sleep(1.0)
        end
    end
end
