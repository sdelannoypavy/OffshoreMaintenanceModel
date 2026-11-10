# heuristic_extraction.jl
#
# Method-agnostic policy extraction ("heuristic_extraction" option of every make_policy_*
# function, on by default). All five methods (fluid_cg, fluid_cg_light, compact,
# fluid_cg_order2, lift_project) build a master LP whose q-variables give, at stage 1 and
# the (known, deterministic) current state/quota, the LP's own belief about which mode
# (maintain or not) each component should be in, and how many free days each pole should
# use. Instead of letting the method-specific machinery (column argmax, or the greedy
# LP-weight-based day picker of extract_policy_compact) also decide the *schedule* --
# which, as fluid_lp_cg.jl's solve_pricing docstring on Δ_pairs and the discussion in
# report_one_pole.txt (scenario 100, period 119) showed, can pick a start day that the
# ADDITIVE proxy sees as free but that costs real, non-additive penalty (a maintained
# component keeps accruing its pre-repair failure penalty for every day the repair is
# delayed; the additive model's `mode == 1 ? δ[m][x] : cc[m]` term does not charge that) --
# this file provides ONE shared rule, used identically by every method:
#
#   1. mode_m := argmax_{mode in {1,2}} (q-mass of component m being in `mode`), read off
#      the method's own q at the current state (marginalised over κ).
#   2. κ_p := as many free days as the remaining quota k0(p) allows, capped at the number
#      of days of actual maintenance of pole p (greedy, optimal under a uniform daily
#      penalty, see extract_policy_heuristic).
#   3. Given the resulting maintained set and κ_p, re-derive an explicit schedule (start
#      days, specific free days) that is optimal for the TRUE, non-additive cost, not the
#      additive proxy: every component picked for maintenance starts on the EARLIEST
#      feasible day of the start-day grid (delaying it can only add pre-repair exposure,
#      never help, since δ_m(x) >= 0 and repair always weakly improves the state), and the
#      κ_p free days are placed on days of actual ongoing maintenance (the only days
#      `pole_penalty`, instances.jl, ever waives on a free day -- a day with no maintenance
#      gets no benefit from being declared free).
#
# This turns the two aggregate LP decisions (mode, κ) -- which are shared, method-agnostic
# information -- into a schedule via a single deterministic rule, instead of leaving the
# schedule to whatever the method's own column pool / LP-weight tie-break happens to
# produce.

"""
    extract_policy_heuristic(fd, x0, k0, mode_mass, kappa_mass)

`mode_mass(m, mode)` and `kappa_mass(p, κ)` must return the method's own q-mass (already
evaluated with `value(...)`, not JuMP variable references) for component `m` being in
`mode ∈ {1, 2}`, respectively for pole `p` using exactly `κ` free days this period, at the
fixed current state `x0` / quota `k0`. Returns a `Column`.
"""
function extract_policy_heuristic(fd::FluidData, x0::Vector{Int}, k0::Vector{Int},
                                  mode_mass::Function, kappa_mass::Function)
    # Free days are used greedily: as many as the remaining quota allows, on the days of
    # actual maintenance (schedule_true_optimal caps κ_p at the number of such days). With a
    # uniform daily penalty this is optimal: a free day saves exactly one daily penalty
    # whenever it is used, and the quota is lost at the end of the year, so keeping it can
    # never save more than using it now. Taking instead the argmax of the LP's κ mass
    # (`kappa_mass`, no longer used) rounded κ independently of the maintenance decision, and
    # could maintain without declaring any free day while quota was left. To be revisited if
    # the daily penalty stops being uniform.
    κ_p = copy(k0)
    maintain = [mode_mass(m, 2) > mode_mass(m, 1) for m in 1:fd.M]
    return schedule_true_optimal(fd, x0, maintain, κ_p)
end

"""
    schedule_true_optimal(fd, x0, maintain, κ_p)

Given which components are maintained this period (`maintain::Vector{Bool}`) and how many
free days each pole uses (`κ_p::Vector{Int}`), build the schedule that minimises the TRUE
period cost (`pole_penalty`, instances.jl):
  - every maintained component starts on the earliest day of the start-day grid
    (`FIRST_START_DAY:QUOTA_STEP:last_start[m]`, see solve_pricing in fluid_lp_cg.jl)
    compatible with the shared technician capacity `inst.R` (a simple earliest-fit greedy,
    most urgent component -- largest current failure penalty `δ_m(x0[m])` -- scheduled
    first);
  - a component the LP wants maintained but that the greedy cannot fit this period (a rare
    rounding artefact of using the LP's own marginal to decide `maintain` independently for
    every component) is simply left unmaintained this period rather than left infeasible;
  - the κ_p[p] free days of pole p are placed on the first κ_p[p] days (in day order --
    they are interchangeable under a uniform daily penalty, see the discussion in
    implemented_formulations.tex) among the days some component of the pole is actually
    under maintenance, since those are the only days `pole_penalty` ever waives.
"""
function schedule_true_optimal(fd::FluidData, x0::Vector{Int}, maintain::Vector{Bool}, κ_p::Vector{Int})
    inst = fd.inst; M = fd.M; T = inst.T_period; K = inst.n_poles
    last_start = [T - fd.d[m] + 1 - inst.margin for m in 1:M]
    first_start = FIRST_START_DAY
    starts_grid(m) = first_start:QUOTA_STEP:last_start[m]

    candidates = [m for m in 1:M if maintain[m] && last_start[m] >= first_start]
    sort!(candidates; by = m -> (-fd.δ[m][x0[m]], -fd.rho[m]))

    start = zeros(Int, M)
    demand = zeros(T)   # shared technician demand: inst.R is shared across every pole

    for m in candidates
        chosen = 0
        for τ in starts_grid(m)
            if all(demand[s] + fd.rho[m] <= inst.R + 1e-9 for s in τ:(τ + fd.d[m] - 1))
                chosen = τ
                break
            end
        end
        chosen == 0 && continue   # could not fit this period: left unmaintained (see docstring)
        start[m] = chosen
        for s in chosen:(chosen + fd.d[m] - 1); demand[s] += fd.rho[m]; end
    end

    free_days = Vector{Vector{Int}}(undef, K)
    for p in 1:K
        ongoing = falses(T)
        for m in fd.comps_of_pole[p]
            start[m] == 0 && continue
            for s in start[m]:(start[m] + fd.d[m] - 1); ongoing[s] = true; end
        end
        ong_days = findall(ongoing)
        free_days[p] = ong_days[1:min(κ_p[p], length(ong_days))]
    end

    return Column(start, free_days)
end
