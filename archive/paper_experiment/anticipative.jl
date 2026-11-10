# anticipative.jl
#
# Cost of the best ANTICIPATIVE policy (perfect information: the degradation of every component
# over the whole horizon is known in advance) on the scenarios of run_experiment.jl, for the
# simplified instances (*_simple). Usage:
#
#   julia --project=. paper_experiment/anticipative.jl <instance> [n_scenarios] [first_scenario]
#
# Writes paper_experiment/results/final_results/<instance>_anticipative_costs.csv
# (one row per scenario: scenario, anticipative_cost, solve_time, mip_gap).
#
# Scenarios: simulate_scenario (simulation.jl) uses common random numbers. Scenario s draws
# U = rand(MersenneTwister(1000 + s), T_total, T_period, M) up front, and on a simplified
# instance component m transitions at the end of period t with U[t, 1, m] if it is not
# maintained (it is reset to the as-new state 1 if it is). The trajectory of a component is
# therefore a deterministic function of its own maintenance periods, and the period cost is
#   Σ_m δ_m(x_m) + Σ_{m maintained} cc_m + column_penalty(schedule),
# exactly as in simulate_period's simplified branch.
#
# MILP (one per scenario, same structure for every scenario, only the arc costs change):
#   * per component m, a path over the periods: arc (i, j), i <= j, means "as new at the start
#     of period i, next maintained in period j" (cost Σ_{t=i}^{j} δ_m(x_t) + cc_m, then as new
#     at period j+1); end arc i means "as new at period i, never maintained again" (cost
#     Σ_{t=i}^{T} δ_m(x_t)). The states x_t along an arc are computed from U.
#   * per period t, the schedule on the start-day grid of fluid_lp_cg.jl's pricing problem
#     (start days FIRST_START_DAY, +QUOTA_STEP, ...; every duration is a multiple of QUOTA_STEP),
#     i.e. on the slots of QUOTA_STEP days covering days FIRST_START_DAY..T_period: start slot
#     variables, technician capacity per slot, and per pole and slot an indicator that the pole
#     is under maintenance (cost QUOTA_STEP days of daily penalty). Restricting the start days
#     to this grid does not change the optimal value (uniform daily penalty, durations multiple
#     of the grid step), see the pricing problem of fluid_lp_cg.jl.

using JuMP, Gurobi, Printf, Random, Statistics

include(joinpath(@__DIR__, "instances.jl"))
include(joinpath(@__DIR__, "fluid_lp_cg.jl"))
include(joinpath(@__DIR__, "simulation.jl"))

"Draws of scenario `scen`, exactly as in simulate_scenario."
scenario_draws(fd::FluidData, scen::Int; seed_base::Int = 1000) =
    rand(MersenneTwister(seed_base + scen), fd.inst.n_years * N_PERIODS_PER_YEAR, fd.inst.T_period, fd.M)

"""
Arc costs of component m: C[i, j] (as new at period i, maintained in period j) and E[i] (as new
at period i, never maintained again), from the draws U.
"""
function arc_costs(fd::FluidData, U, m::Int, T::Int)
    P = fd.inst.poles[fd.pole_of[m]].P_period[fd.comp_of[m]]   # same matrix as simulate_period
    C = fill(Inf, T, T); E = zeros(T)
    for i in 1:T
        x = 1; cum = 0.0
        for t in i:T
            cum += fd.δ[m][x]
            C[i, t] = cum + fd.cc[m]
            x = sample_next(P, x, U[t, 1, m])
        end
        E[i] = cum
    end
    return C, E
end

struct AnticipativeModel
    model::Model
    arc::Dict{NTuple{3,Int},VariableRef}   # (m, i, j)
    endarc::Dict{Tuple{Int,Int},VariableRef}   # (m, i)
    start::Dict{NTuple{3,Int},VariableRef}  # (m, t, slot): maintenance of m in period t starts at `slot`
    T::Int
end

function build_anticipative(fd::FluidData)
    inst = fd.inst
    @assert inst.simplified "anticipative.jl is for the simplified instances"
    @assert all(K0 == 0 for K0 in fd.K0) "no penalty-free days expected on the simplified instances"
    @assert all(all(==(inst.poles[1].daily_penalty[1]), p.daily_penalty) for p in inst.poles) "uniform daily penalty expected"
    T = inst.n_years * N_PERIODS_PER_YEAR
    Tp = inst.T_period; Δ = QUOTA_STEP; first = FIRST_START_DAY
    @assert all(d % Δ == 0 for d in fd.d) && (Tp - first + 1) % Δ == 0
    nslots = (Tp - first + 1) ÷ Δ
    pen = inst.poles[1].daily_penalty[1]

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    arc = Dict{NTuple{3,Int},VariableRef}(); endarc = Dict{Tuple{Int,Int},VariableRef}()
    start = Dict{NTuple{3,Int},VariableRef}()
    for m in 1:fd.M
        for i in 1:T
            for j in i:T
                arc[(m, i, j)] = @variable(model, binary = true)
            end
            endarc[(m, i)] = @variable(model, binary = true)
        end
        # path from node 1 to the end: node i is entered at period 1 (i = 1) or after a maintenance in period i-1
        for i in 1:T
            inflow = i == 1 ? 1.0 : sum(arc[(m, ip, i - 1)] for ip in 1:(i - 1))
            @constraint(model, sum(arc[(m, i, j)] for j in i:T) + endarc[(m, i)] == inflow)
        end
    end
    obj = AffExpr(0.0)
    for t in 1:T
        cover = Dict{Tuple{Int,Int},AffExpr}()   # (m, slot) -> under repair during slot
        for m in 1:fd.M
            nd = fd.d[m] ÷ Δ
            slots = 1:(nslots - nd + 1)
            for k in slots
                start[(m, t, k)] = @variable(model, binary = true)
                for kk in k:(k + nd - 1)
                    add_to_expression!(get!(() -> AffExpr(0.0), cover, (m, kk)), start[(m, t, k)])
                end
            end
            # maintained in period t <=> some arc ends in t
            @constraint(model, sum(start[(m, t, k)] for k in slots) == sum(arc[(m, i, t)] for i in 1:t))
        end
        for kk in 1:nslots
            @constraint(model, sum(fd.rho[m] * get(cover, (m, kk), AffExpr(0.0)) for m in 1:fd.M) <= inst.R)
            for p in 1:inst.n_poles
                u = @variable(model, lower_bound = 0, upper_bound = 1)
                for m in fd.comps_of_pole[p]
                    haskey(cover, (m, kk)) && @constraint(model, u >= cover[(m, kk)])
                end
                add_to_expression!(obj, pen * Δ, u)
            end
        end
    end
    @objective(model, Min, obj)
    return AnticipativeModel(model, arc, endarc, start, T)
end

"Set the arc costs of scenario `scen` and solve. Returns (cost, solve time, relative MIP gap)."
function solve_anticipative!(am::AnticipativeModel, fd::FluidData, scen::Int)
    U = scenario_draws(fd, scen)
    for m in 1:fd.M
        C, E = arc_costs(fd, U, m, am.T)
        for i in 1:am.T
            for j in i:am.T
                set_objective_coefficient(am.model, am.arc[(m, i, j)], C[i, j])
            end
            set_objective_coefficient(am.model, am.endarc[(m, i)], E[i])
        end
    end
    optimize!(am.model)
    @assert termination_status(am.model) == MOI.OPTIMAL "scenario $scen: $(termination_status(am.model))"
    return objective_value(am.model), solve_time(am.model), relative_gap(am.model)
end

"""
Check the cost model against simulate_scenario: a simple feasible policy (maintain, in every
pole, its most degraded component if it is not new, at the first slot) is simulated by
simulation.jl, and its cost is recomputed from the arc costs and the slot model.
"""
function check_cost_model(fd::FluidData, scen::Int)
    Tp = fd.inst.T_period; first = FIRST_START_DAY
    function policy(x, k, t)
        start = zeros(Int, fd.M)
        for p in 1:fd.inst.n_poles
            ms = fd.comps_of_pole[p]
            m = ms[argmax([x[m] for m in ms])]
            x[m] > 1 && fd.rho[m] <= fd.inst.R / fd.inst.n_poles + 1e-9 && (start[m] = first)
        end
        return Column(start, [Int[] for _ in 1:fd.inst.n_poles]), 0.0
    end
    sim = simulate_scenario(fd, policy, scen; method = "check", instance = fd.inst.name, periods_file = nothing)
    # recompute: replay the policy on the draws, cost from arc costs + one slot per maintenance
    U = scenario_draws(fd, scen); T = fd.inst.n_years * N_PERIODS_PER_YEAR
    x = ones(Int, fd.M); total = 0.0
    for t in 1:T
        col, _ = policy(copy(x), fd.K0, t)
        total += sum(fd.δ[m][x[m]] for m in 1:fd.M) + sum(fd.cc[m] for m in 1:fd.M if col.start[m] > 0; init = 0.0)
        for p in 1:fd.inst.n_poles
            ds = [fd.d[m] for m in fd.comps_of_pole[p] if col.start[m] > 0]
            total += isempty(ds) ? 0.0 : maximum(ds) * fd.inst.poles[p].daily_penalty[1]
        end
        for m in 1:fd.M
            x[m] = col.start[m] > 0 ? 1 : sample_next(fd.inst.poles[fd.pole_of[m]].P_period[fd.comp_of[m]], x[m], U[t, 1, m])
        end
    end
    return sim.total, total
end

if abspath(PROGRAM_FILE) == @__FILE__
    instance = ARGS[1]
    n_scen = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1000
    first_scen = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    fd = build_fluid_data(only(filter(i -> i.name == instance, instances)))
    a, b = check_cost_model(fd, 1)
    @printf("cost model check (scenario 1): simulate_scenario = %.4f, recomputed = %.4f\n", a, b)
    @assert abs(a - b) < 1e-6
    out = joinpath(@__DIR__, "results", "final_results", "$(instance)_anticipative_costs.csv")
    done = Set{Int}()
    isfile(out) && for l in readlines(out)[2:end]; push!(done, parse(Int, split(l, ",")[1])); end
    t0 = time(); am = build_anticipative(fd); @printf("model built in %.1fs\n", time() - t0)
    for scen in first_scen:(first_scen + n_scen - 1)
        scen in done && continue
        c, st, gap = solve_anticipative!(am, fd, scen)
        append_csv(out, "scenario,anticipative_cost,solve_time,mip_gap", @sprintf("%d,%.4f,%.3f,%.2e", scen, c, st, gap))
        @printf("  scenario %4d  anticipative cost = %.4f  (%.2fs)\n", scen, c, st)
    end
end
