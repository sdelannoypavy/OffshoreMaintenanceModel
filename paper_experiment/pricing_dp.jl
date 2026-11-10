# pricing_dp.jl
#
# Exact pricing problems of the Joint fluid LP (eq:pricing-first and eq:pricing of paper.tex) by
# dynamic programming over the poles, instead of the MILP of fluid_cg_paper.jl (build_pricing_core).
#
# The poles are only coupled by the technician capacity, which is checked on the days
# FIRST_START_DAY, FIRST_START_DAY + QUOTA_STEP, ... (as in the MILP). Every other term of the
# reduced cost is local to a pole: the joint action cost f(b) and the true cost c⊗(x0, b) of a pole
# only depend on the repairs and the declared days of this pole, and the dual terms only depend on
# the local actions (δ_m, κ_p) of its components. Hence
#   1. for every pole, the local repair schedules that satisfy the technician capacity on their own
#      are enumerated once (start day of each component, on the grid of start days of the MILP),
#      with the load of technicians they induce on the check days;
#   2. for a schedule and a number κ of declared days, the best declared days are the κ days of
#      largest penalty (greedy), and the best κ is found by enumeration;
#   3. the poles are combined by dynamic programming, the state being the load of technicians
#      accumulated on the check days (and whether some component is maintained, for the
#      mobilization cost).
# The result is exact. Its cost is polynomial in the number of poles, but exponential in the number
# of check days, so that it requires the start days to be discretised (QUOTA_STEP = 15: 3 check
# days). The declared days are not restricted to blocks, so that κ does not need to be a multiple of
# QUOTA_STEP.
#
# Only the duals of the Joint fluid LP are supported (no order-2 pairs, no lift-and-project cuts).

"Local repair schedule of a pole: start day of each of its components (0 if not maintained) and load of technicians on the check days."
struct PoleSchedule
    start::Vector{Int}          # start day of the components of the pole, in the order of fd.comps_of_pole[p]
    load::Vector{Float64}       # load of technicians of the pole on each check day
end

"Days on which the technician capacity is checked (as in build_pricing_core)."
check_days(fd::FluidData) = collect(FIRST_START_DAY:QUOTA_STEP:fd.inst.T_period)

"Start days of component m on the grid of the pricing problems (as in build_pricing_core)."
function grid_starts(fd::FluidData, m::Int)
    last_start = fd.inst.T_period - fd.d[m] + 1 - fd.inst.margin
    return last_start >= FIRST_START_DAY ? collect(FIRST_START_DAY:QUOTA_STEP:last_start) : Int[]
end

const POLE_SCHEDULES_CACHE = Dict{Tuple{UInt,Int,Int,Int},Vector{PoleSchedule}}()

"Local repair schedules of pole p that satisfy the technician capacity (cached)."
function pole_schedules(fd::FluidData, p::Int)
    get!(POLE_SCHEDULES_CACHE, (objectid(fd), p, QUOTA_STEP, FIRST_START_DAY)) do
        comps = fd.comps_of_pole[p]; days = check_days(fd); R = fd.inst.R
        out = PoleSchedule[]
        start = zeros(Int, length(comps)); load = zeros(length(days))
        function rec(i)
            if i > length(comps)
                push!(out, PoleSchedule(copy(start), copy(load)))
                return
            end
            start[i] = 0; rec(i + 1)                       # component not maintained
            m = comps[i]
            for s in grid_starts(fd, m)
                cover = [c for (c, τ) in enumerate(days) if s <= τ <= s + fd.d[m] - 1]
                all(load[c] + fd.rho[m] <= R + 1e-9 for c in cover) || continue
                for c in cover; load[c] += fd.rho[m]; end
                start[i] = s; rec(i + 1)
                for c in cover; load[c] -= fd.rho[m]; end
            end
            start[i] = 0
        end
        rec(1)
        out
    end
end

"""
Penalty per day of pole p for the local schedule `sch`, before any day is declared penalty-free:
f(b) if `x0` is nothing (penalty of the maintenance days), the penalty of the true cost c⊗(x0, b)
otherwise.
"""
function pole_day_costs(fd::FluidData, p::Int, sch::PoleSchedule, x0::Union{Nothing,Vector{Int}})
    comps = fd.comps_of_pole[p]; T = fd.inst.T_period; pen = fd.inst.poles[p].daily_penalty
    c = zeros(T)
    for τ in 1:T
        ongoing = any(sch.start[i] > 0 && sch.start[i] <= τ <= sch.start[i] + fd.d[m] - 1 for (i, m) in enumerate(comps))
        if ongoing
            c[τ] = pen[τ]
        elseif !isnothing(x0)
            c[τ] = pen[τ] * maximum(capacity_loss(fd, m, (sch.start[i] > 0 && τ >= sch.start[i] + fd.d[m]) ? 1 : x0[m])
                                    for (i, m) in enumerate(comps))
        end
    end
    return c
end

"Cost of a local schedule before the dual terms: cost if no day is declared, saving of the κ best declared days, and these days in decreasing order of penalty."
struct ScheduleCost
    base::Float64               # replacement costs (first step) + penalty of the pole with no declared day
    prefix::Vector{Float64}     # prefix[κ] = penalty saved by declaring the κ most expensive days
    order::Vector{Int}          # days in decreasing order of penalty
end

const SCHEDULE_COSTS_CACHE = Dict{Any,Vector{ScheduleCost}}()

"""
Costs of the local schedules of pole p (independent of the duals, cached): f(b) if `x0` is nothing,
the true cost c⊗(x0, b) otherwise, which only depends on the degradation states of the components of p.
"""
function schedule_costs(fd::FluidData, p::Int, x0::Union{Nothing,Vector{Int}})
    comps = fd.comps_of_pole[p]
    key = (objectid(fd), p, QUOTA_STEP, FIRST_START_DAY, isnothing(x0) ? nothing : Tuple(x0[comps]))
    get!(SCHEDULE_COSTS_CACHE, key) do
        map(pole_schedules(fd, p)) do sch
            c = pole_day_costs(fd, p, sch, x0)
            order = sortperm(c; rev = true)             # the κ most expensive days are declared
            rep = isnothing(x0) ? 0.0 : sum((fd.cc[m] for (i, m) in enumerate(comps) if sch.start[i] > 0); init = 0.0)
            ScheduleCost(rep + sum(c), cumsum(c[order]), order)
        end
    end
end

"""
Best number of declared days κ ∈ `kappas` of pole p for the schedule `sch` of cost `sc`, with the
dual terms `dtab[(m, mode, κ)]`: (value, κ, declared days).
"""
function best_kappa(fd::FluidData, p::Int, sch::PoleSchedule, sc::ScheduleCost, kappas, dtab)
    comps = fd.comps_of_pole[p]
    best = (Inf, -1)
    for κ in kappas
        val = sc.base - (κ == 0 ? 0.0 : sc.prefix[κ])
        for (i, m) in enumerate(comps)
            val += dtab[(m, sch.start[i] > 0 ? 2 : 1, κ)]
        end
        val < best[1] && (best = (val, κ))
    end
    κ = best[2]
    return best[1], κ, sort(sc.order[1:κ])
end

"""
Pricing by dynamic programming over the poles: minimum over the joint actions of the reduced cost
(without the constant η of the first step), with the dual term `dualf(m, mode, κ)`. Returns the best
column, its value, and the best column of every other final state of value < `pool_tol` (at most
`n_pool - 1` of them).
"""
function solve_pricing_dp(fd::FluidData, kcap::Vector{Int}, x0, dualf; n_pool::Int = 1, pool_tol::Float64 = -1e-6)
    K = fd.inst.n_poles; R = fd.inst.R
    key(load) = Tuple(round.(Int, load .* 1000))
    # dual terms of the local actions, computed once
    dtab = Dict((m, mode, κ) => dualf(m, mode, κ) for p in 1:K for m in fd.comps_of_pole[p] for mode in 1:2 for κ in qrange(kcap[p]))
    # best (value, κ, declared days) of every schedule of every pole: independent of the other poles
    poles = map(1:K) do p
        scs = schedule_costs(fd, p, x0)
        [(sch, best_kappa(fd, p, sch, scs[i], qrange(kcap[p]), dtab)...) for (i, sch) in enumerate(pole_schedules(fd, p))]
    end
    # states: (load key, some component maintained) -> (value, load, choices)
    nd = length(check_days(fd))
    states = Dict{Tuple{Tuple,Bool},Tuple{Float64,Vector{Float64},Vector{Int}}}(
        (key(zeros(nd)), false) => (0.0, zeros(nd), Int[]))
    for p in 1:K
        # dominance: among the schedules of p with the same load and the same flag, only the best one matters
        bestp = Dict{Tuple{Tuple,Bool},Int}()
        for (i, (sch, v, _, _)) in enumerate(poles[p])
            k = (key(sch.load), any(>(0), sch.start))
            (haskey(bestp, k) && poles[p][bestp[k]][2] <= v) || (bestp[k] = i)
        end
        new = Dict{Tuple{Tuple,Bool},Tuple{Float64,Vector{Float64},Vector{Int}}}()
        for ((_, flag), (val, load, ch)) in states, i in values(bestp)
            sch, v, _, _ = poles[p][i]
            nl = load .+ sch.load
            all(nl .<= R + 1e-9) || continue
            k = (key(nl), flag || any(>(0), sch.start))
            nv = val + v
            (haskey(new, k) && new[k][1] <= nv) && continue
            new[k] = (nv, nl, vcat(ch, i))
        end
        states = new
    end
    finals = sort([(val + (flag && MOBILIZATION_COST > 0 ? MOBILIZATION_COST : 0.0), ch)
                   for ((_, flag), (val, _, ch)) in states]; by = first)
    function column(ch)
        start = zeros(Int, fd.M); free_days = Vector{Vector{Int}}(undef, K)
        for p in 1:K
            sch, _, _, days = poles[p][ch[p]]
            for (i, m) in enumerate(fd.comps_of_pole[p]); start[m] = sch.start[i]; end
            free_days[p] = days
        end
        return Column(start, free_days)
    end
    best_val, best_ch = finals[1]
    extras = [column(ch) for (v, ch) in finals[2:min(n_pool, length(finals))] if v < pool_tol]
    return column(best_ch), best_val, extras
end

"First-step pricing problem (eq:pricing-first) by dynamic programming: same output as solve_pricing_first_paper."
function solve_pricing_first_dp(fd::FluidData, master::PaperMaster, du::PaperDuals; n_pool::Int = 1, tol::Float64 = 1e-6, kwargs...)
    (isempty(master.pairs) && isempty(master.cuts)) || error("pricing_dp: order 2 and cuts are not supported")
    x0 = master.x0; k0 = master.k0
    function dualf(m, mode, κ)
        master.H >= 2 || return 0.0
        p = fd.pole_of[m]
        k_next = quota_next(1, master.t0_cyc, k0[p], κ, fd.K0[p])
        return sum(get(du.ν, (m, n, k_next), 0.0) * trans_prob(fd, m, n, x0[m], mode) for n in 1:fd.n_states[m])
    end
    col, val, extras = solve_pricing_dp(fd, k0, x0, dualf; n_pool = n_pool, pool_tol = du.η - tol)
    return col, val - du.η, extras
end

"Pricing problem (eq:pricing) at stage t ≥ 2 by dynamic programming: same output as solve_pricing_t_paper."
function solve_pricing_t_dp(fd::FluidData, master::PaperMaster, du::PaperDuals, t::Int; n_pool::Int = 1, tol::Float64 = 1e-6, kwargs...)
    (isempty(master.pairs) && isempty(master.cuts)) || error("pricing_dp: order 2 and cuts are not supported")
    dualf(m, mode, κ) = get(du.λ, (m, mode, κ, t), 0.0)
    return solve_pricing_dp(fd, fd.K0, nothing, dualf; n_pool = n_pool, pool_tol = -tol)
end
