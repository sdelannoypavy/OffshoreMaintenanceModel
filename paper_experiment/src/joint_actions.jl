# joint_actions.jl
#
# Enumeration of the joint actions, used by JFLP (enum.) and JFLP-Q (jflp_enum.jl) and by the
# two-step baseline (two_step_baseline.jl).
#
# A joint action is a schedule (start day of every repair, on the grid of start_days, model.jl) and
# a set of declared days for every substation. The schedules that satisfy the technician capacity are
# enumerated once per instance. For a schedule and a number κ of declared days of a substation, the
# best declared days are its κ days of largest penalty, so that only these are kept.

"""
Joint actions of an instance, grouped by their local actions:
  - `scheds`: every schedule that satisfies the technician capacity,
  - `sets`: every set of components that can be maintained in the same period, and the set of each schedule,
  - `kappas`: every vector (κ^s)_s of numbers of declared days,
  - for t ≥ 2, one column per (set, κ vector), with the schedule of least joint action cost f(b).
"""
struct JointActionEnum
    scheds::Vector{Vector{Int}}
    sched_set::Vector{Int}          # index in `sets` of the set of components maintained by each schedule
    sets::Vector{BitVector}
    kappas::Vector{Vector{Int}}     # κ^s ∈ qrange(K0^s)
    col_set::Vector{Int}            # columns of the stages t ≥ 2: set,
    col_kappa::Vector{Int}          #   index in `kappas`,
    col_sched::Vector{Int}          #   schedule of least cost,
    col_cost::Vector{Float64}       #   and this cost f(b)
    time::Float64                   # computing time of the enumeration
end

"Every schedule that satisfies the technician capacity on every day."
function enumerate_schedules(fd::FluidData)
    T = fd.inst.T_period
    scheds = Vector{Vector{Int}}()
    load = zeros(T); cur = zeros(Int, fd.M)
    function rec(m)
        if m > fd.M
            push!(scheds, copy(cur)); return
        end
        rec(m + 1)   # m not maintained
        for s in start_days(fd, m)
            days = s:(s + fd.d[m] - 1)
            all(load[τ] + fd.rho[m] <= fd.inst.R + 1e-9 for τ in days) || continue
            load[days] .+= fd.rho[m]; cur[m] = s
            rec(m + 1)
            load[days] .-= fd.rho[m]; cur[m] = 0
        end
    end
    rec(1)
    return scheds
end

"Penalty Σ_{τ ∉ Free} w_τ of the best set Free of κ declared days (the κ days of largest w_τ), for every κ of `κs`."
function best_free_costs(w::Vector{Float64}, κs)
    cs = cumsum(sort(w; rev = true)); tot = sum(w)
    return [κ == 0 ? tot : tot - cs[κ] for κ in κs]
end

"The κ days of largest w_τ (ties broken by the earliest day)."
best_free_days(w::Vector{Float64}, κ::Int) = sort(sortperm(w; rev = true, alg = Base.Sort.DEFAULT_STABLE)[1:κ])

"π_τ m^s_τ: penalty of substation p on every day due to its maintenance (joint action cost f(b) before the declared days)."
function maintenance_weights(fd::FluidData, start::Vector{Int}, p::Int)
    pen = fd.inst.poles[p].daily_penalty
    return [any(under_repair(fd, start, m, τ) for m in fd.comps_of_pole[p]) ? pen[τ] : 0.0 for τ in 1:fd.inst.T_period]
end

"π_τ ℓ^s_τ: penalty of substation p on every day in the full model, for the states x0 (true cost before the declared days)."
function loss_weights(fd::FluidData, x0::Vector{Int}, start::Vector{Int}, p::Int)
    pen = fd.inst.poles[p].daily_penalty; ms = fd.comps_of_pole[p]
    return [any(under_repair(fd, start, m, τ) for m in ms) ? pen[τ] :
            pen[τ] * maximum(capacity_loss(fd, m, (start[m] > 0 && τ >= start[m] + fd.d[m]) ? 1 : x0[m]) for m in ms)
            for τ in 1:fd.inst.T_period]
end

"""
Penalty of substation p on every day of a period starting in the states x0, before the declared
days: `loss_weights` in the full model, `maintenance_weights` in the simplified model, whose failure
costs do not depend on the schedule (they are added by `first_step_constant`).
"""
period_weights(fd::FluidData, x0::Vector{Int}, start::Vector{Int}, p::Int) =
    fd.inst.simplified ? maintenance_weights(fd, start, p) : loss_weights(fd, x0, start, p)

"Part of the cost of a period that does not depend on the joint action: the failure costs in the simplified model, 0 in the full model."
first_step_constant(fd::FluidData, x0::Vector{Int}) =
    fd.inst.simplified ? sum(failure_cost(fd, m, x0[m]) for m in 1:fd.M) : 0.0

kappa_index(κ::Int) = κ ÷ STEP + 1

const ENUM_CACHE = Dict{UInt,JointActionEnum}()

"Enumeration of the joint actions of `fd`, computed once and kept in memory."
joint_action_enum(fd::FluidData) = get!(() -> build_joint_action_enum(fd), ENUM_CACHE, objectid(fd))

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
        costs = [best_free_costs(maintenance_weights(fd, s, p), qrange(fd.K0[p])) for p in 1:K]
        for (j, κ) in enumerate(kappas)
            c = sum(costs[p][kappa_index(κ[p])] for p in 1:K)
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
First-step columns b ∈ B(x0): every (set, κ vector) with κ^s ≤ k0^s, with its least cost c⊗(x0, b)
over its schedules (true cost of the period). Returns the vectors (set, κ index, schedule, cost).
"""
function first_step_enum(fd::FluidData, E::JointActionEnum, x0::Vector{Int}, k0::Vector{Int})
    K = fd.inst.n_poles
    feas = [j for (j, κ) in enumerate(E.kappas) if all(κ[p] <= k0[p] for p in 1:K)]
    base = first_step_constant(fd, x0)
    best = fill(Inf, length(E.sets), length(feas)); arg = zeros(Int, length(E.sets), length(feas))
    for (i, s) in enumerate(E.scheds)
        repl = sum(fd.cc[m] for m in 1:fd.M if s[m] > 0; init = 0.0)
        costs = [best_free_costs(period_weights(fd, x0, s, p), qrange(fd.K0[p])) for p in 1:K]
        for (jj, j) in enumerate(feas)
            κ = E.kappas[j]
            c = base + repl + sum(costs[p][kappa_index(κ[p])] for p in 1:K)
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

"Joint action of a first-step column: its schedule, and the κ^s days of largest penalty of every substation."
function first_step_column(fd::FluidData, E::JointActionEnum, x0::Vector{Int}, sched::Int, j::Int)
    s = E.scheds[sched]; κ = E.kappas[j]
    return Column(copy(s), [best_free_days(period_weights(fd, x0, s, p), κ[p]) for p in 1:fd.inst.n_poles])
end
