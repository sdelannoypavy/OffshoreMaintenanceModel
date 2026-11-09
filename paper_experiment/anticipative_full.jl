# anticipative_full.jl
#
# Cost of the best ANTICIPATIVE policy (perfect information: the degradation of every component over
# the whole horizon is known in advance) on the scenarios of the simulations, for the model of
# paper.tex (true cost Tc with the maximum of the capacity losses of a pole, penalty-free days and
# annual quota), one MILP per scenario. Usage:
#
#   julia --project=. paper_experiment/anticipative_full.jl <instance> <first_scenario> <last_scenario> <out.csv>
#
# Same environment variables as the simulations (QUOTA_STEP, MTBF_SCALE, ...). Writes one row per
# scenario: scenario, anticipative_cost, solve_time, mip_gap.
#
# Scenarios: simulate_scenario (simulation.jl) draws U = rand(MersenneTwister(1000 + s), T, T_period,
# M) up front, and component m, if it is not maintained in period t, moves at the end of the period
# with U[t, 1, m] and P_period (it is reset to the new state 1 if it is maintained). The trajectory of
# a component is therefore a deterministic function of its maintenance periods.
#
# MILP:
#   * per component m, a path over the periods: arc (i, j), i ≤ j, means "new at the start of period
#     i, next maintained in period j". y[m, t, i] ∈ [0, 1] is 1 iff component m is new at period i and
#     not maintained in i..t-1, so that its state in period t is s_m(i, t), computed from U.
#   * per period, the start days are on the grid FIRST_START_DAY, FIRST_START_DAY + Δ, ... (Δ =
#     QUOTA_STEP, every duration is a multiple of Δ), as in the simulations. The days of the period
#     are split in blocks: the lead time before FIRST_START_DAY (no maintenance), the slots of Δ days
#     of the grid, and the
#     remaining last days (no maintenance, every repair is complete). On every block b, the capacity
#     loss of a pole is Cl_b = 1 if one of its components is under repair, else the largest loss
#     ℓ_m(n_m) of its components whose repair is not complete (eq:intro-capacity-loss).
#   * f_b declared penalty-free days in block b, Σ_b f_b = κ_t = Δ q_t (multiple of Δ, as in the
#     simulations), Σ_{t in a year} κ_t ≤ K0. The penalty of block b is π (|b| Cl_b − f_b Cl_b), the
#     product f_b Cl_b being written with the binary expansion of f_b (exact).
#   * technician capacity on every slot, replacement cost Rc_m and mobilization cost.

using JuMP, Gurobi, Printf, Random, Statistics

include(joinpath(@__DIR__, "instances.jl"))
include(joinpath(@__DIR__, "fluid_lp_cg.jl"))
include(joinpath(@__DIR__, "simulation.jl"))

"Draws of scenario `scen`, exactly as in simulate_scenario."
scenario_draws(fd::FluidData, scen::Int; seed_base::Int = 1000) =
    rand(MersenneTwister(seed_base + scen), fd.inst.n_years * N_PERIODS_PER_YEAR, fd.inst.T_period, fd.M)

"s[i, t]: state of component m in period t if it is new at period i and not maintained in i..t-1 (t ≥ i)."
function arc_states(fd::FluidData, U, m::Int, T::Int)
    P = fd.inst.poles[fd.pole_of[m]].P_period[fd.comp_of[m]]   # same matrix as simulate_period
    s = zeros(Int, T, T)
    for i in 1:T
        x = 1
        for t in i:T
            s[i, t] = x
            x = sample_next(P, x, U[t, 1, m])
        end
    end
    return s
end

"Blocks of days of a period: (first day, last day, slot index or 0 if no maintenance can take place)."
function day_blocks(fd::FluidData)
    Tp = fd.inst.T_period; F = FIRST_START_DAY; Δ = QUOTA_STEP
    nslots = (Tp - F + 1) ÷ Δ
    blocks = Tuple{Int,Int,Int}[]
    F > 1 && push!(blocks, (1, F - 1, 0))
    for k in 1:nslots
        push!(blocks, (F + (k - 1) * Δ, F + k * Δ - 1, k))
    end
    F + nslots * Δ <= Tp && push!(blocks, (F + nslots * Δ, Tp, 0))
    return blocks, nslots
end

struct AnticipativeFull
    model::Model
    arc::Dict{NTuple{3,Int},VariableRef}     # (m, i, j)
    start::Dict{NTuple{3,Int},VariableRef}   # (m, t, slot)
    q::Matrix{VariableRef}                   # κ_t / Δ of every pole (p, t)
end

function build_anticipative_full(fd::FluidData, scen::Int)
    inst = fd.inst; T = inst.n_years * N_PERIODS_PER_YEAR; Δ = QUOTA_STEP; M = fd.M
    π = constant_daily_pen(fd)
    @assert all(d % Δ == 0 for d in fd.d) "every repair duration must be a multiple of QUOTA_STEP"
    @assert inst.margin == 0
    blocks, nslots = day_blocks(fd)
    U = scenario_draws(fd, scen)
    S = [arc_states(fd, U, m, T) for m in 1:M]
    ℓ(m, n) = capacity_loss(fd, m, n)

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    set_optimizer_attribute(model, "MIPGap", 1e-6)
    arc = Dict{NTuple{3,Int},VariableRef}(); y = Dict{NTuple{3,Int},VariableRef}()
    start = Dict{NTuple{3,Int},VariableRef}()
    obj = AffExpr(0.0)
    for m in 1:M
        for i in 1:T, j in i:T
            arc[(m, i, j)] = @variable(model, binary = true)
        end
        for i in 1:T
            inflow = i == 1 ? AffExpr(1.0) : sum(arc[(m, ip, i - 1)] for ip in 1:(i - 1))
            y[(m, i, i)] = @variable(model, lower_bound = 0, upper_bound = 1)
            @constraint(model, y[(m, i, i)] == inflow)
            for t in (i + 1):T
                y[(m, t, i)] = @variable(model, lower_bound = 0, upper_bound = 1)
                @constraint(model, y[(m, t, i)] == y[(m, t - 1, i)] - arc[(m, i, t - 1)])
            end
            @constraint(model, y[(m, T, i)] - arc[(m, i, T)] >= 0)   # never maintained again after T
        end
    end
    q = [@variable(model, integer = true, lower_bound = 0, upper_bound = fd.K0[p] ÷ Δ) for p in 1:inst.n_poles, t in 1:T]
    for t in 1:T
        # maintenance decisions and start slots
        maint = Dict{Int,AffExpr}(); cover = Dict{Tuple{Int,Int},AffExpr}(); done = Dict{Tuple{Int,Int},AffExpr}()
        for m in 1:M
            nd = fd.d[m] ÷ Δ
            slots = 1:(nslots - nd + 1)
            for k in slots
                start[(m, t, k)] = @variable(model, binary = true)
                for kk in k:(k + nd - 1)
                    add_to_expression!(get!(() -> AffExpr(0.0), cover, (m, kk)), start[(m, t, k)])
                end
            end
            maint[m] = sum(start[(m, t, k)] for k in slots)
            @constraint(model, maint[m] == sum(arc[(m, i, t)] for i in 1:t))
            add_to_expression!(obj, fd.cc[m], maint[m])
            for (bi, (_, _, k)) in enumerate(blocks)
                # repair of m complete before block bi: started at a slot ending before it
                done[(m, bi)] = k == 0 ? (bi == 1 ? AffExpr(0.0) : 1.0 * maint[m]) :
                                sum((start[(m, t, kk)] for kk in slots if kk + nd - 1 < k); init = AffExpr(0.0))
            end
        end
        for k in 1:nslots   # technician capacity
            @constraint(model, sum(fd.rho[m] * get(cover, (m, k), AffExpr(0.0)) for m in 1:M) <= inst.R)
        end
        if MOBILIZATION_COST > 0
            mob = @variable(model, lower_bound = 0, upper_bound = 1)
            for m in 1:M; @constraint(model, mob >= maint[m]); end
            add_to_expression!(obj, MOBILIZATION_COST, mob)
        end
        # capacity loss, penalty-free days and penalty of every pole and block
        fsum = AffExpr(0.0)
        for p in 1:inst.n_poles
            ms = fd.comps_of_pole[p]
            Lm = Dict(m => sum((ℓ(m, S[m][i, t]) * y[(m, t, i)] for i in 1:t if ℓ(m, S[m][i, t]) > 0); init = AffExpr(0.0)) for m in ms)
            fp = AffExpr(0.0)
            for (bi, (d1, d2, k)) in enumerate(blocks)
                len = d2 - d1 + 1
                Cl = @variable(model, lower_bound = 0, upper_bound = 1)
                if k > 0   # maintenance day of the pole
                    for m in ms
                        haskey(cover, (m, k)) && @constraint(model, Cl >= cover[(m, k)])
                    end
                end
                for m in ms   # loss of the components whose repair is not complete (ℓ ≤ 1)
                    isempty(Lm[m].terms) || @constraint(model, Cl >= Lm[m] - done[(m, bi)])
                end
                # f free days in the block, binary expansion, w = f Cl
                nb = floor(Int, log2(len)) + 1
                bits = [@variable(model, binary = true) for _ in 1:nb]
                f = sum(2^(i - 1) * bits[i] for i in 1:nb)
                @constraint(model, f <= len)
                z = [@variable(model, lower_bound = 0) for _ in 1:nb]
                for i in 1:nb
                    @constraint(model, z[i] <= bits[i]); @constraint(model, z[i] <= Cl)
                end
                add_to_expression!(obj, π * len, Cl)
                add_to_expression!(obj, -π, sum(2^(i - 1) * z[i] for i in 1:nb))
                add_to_expression!(fp, f)
            end
            @constraint(model, fp == Δ * q[p, t])   # κ_t of the pole
        end
    end
    for p in 1:inst.n_poles, y0 in 1:inst.n_years   # annual quota of every pole
        ts = ((y0 - 1) * N_PERIODS_PER_YEAR + 1):(y0 * N_PERIODS_PER_YEAR)
        @constraint(model, Δ * sum(q[p, t] for t in ts) <= fd.K0[p])
    end
    @objective(model, Min, obj)
    return AnticipativeFull(model, arc, start, q)
end

"Constant daily penalty of the instance."
function constant_daily_pen(fd::FluidData)
    pens = unique(vcat([p.daily_penalty for p in fd.inst.poles]...))
    length(pens) == 1 || error("constant daily penalty expected")
    return pens[1]
end

"""
Fix the decisions of a simulated policy (periods file of simulate_scenario) in the MILP of scenario
`scen`: its cost must then be the simulated cost (check of the cost model).
"""
function fix_policy!(am::AnticipativeFull, fd::FluidData, rows)
    for r in rows
        t = parse(Int, r[4])
        maintained = isempty(r[7]) ? Int[] : parse.(Int, split(r[7], "-"))
        days = isempty(r[8]) ? Int[] : parse.(Int, split(r[8], "-"))
        for ((m, tt, k), v) in am.start
            tt == t || continue
            i = findfirst(==(m), maintained)
            fix(v, (!isnothing(i) && FIRST_START_DAY + (k - 1) * QUOTA_STEP == days[i]) ? 1.0 : 0.0; force = true)
        end
        for (p, κ) in enumerate(parse.(Int, split(r[9], "-")))
            fix(am.q[p, t], κ / QUOTA_STEP; force = true)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    instance = ARGS[1]; first_scen = parse(Int, ARGS[2]); last_scen = parse(Int, ARGS[3]); out = ARGS[4]
    haskey(ENV, "GUROBI_THREADS") && Gurobi.GRBsetintparam(GRB_ENV, "Threads", parse(Int, ENV["GUROBI_THREADS"]))
    fd = build_fluid_data(only(filter(i -> i.name == instance, instances)))
    # optional check: the decisions of a simulated policy, fixed in the MILP, give the simulated cost
    if length(ARGS) >= 5
        rows = [split(l, ",") for l in readlines(ARGS[5])[2:end]]
        for s in first_scen:min(first_scen + 2, last_scen)
            rs = [r for r in rows if parse(Int, r[3]) == s]
            length(rs) == fd.inst.n_years * N_PERIODS_PER_YEAR || continue
            am = build_anticipative_full(fd, s); fix_policy!(am, fd, rs); optimize!(am.model)
            sim = sum(parse(Float64, r[10]) + parse(Float64, r[11]) for r in rs)
            @printf("check scenario %d: simulated cost %.4f, MILP with the fixed decisions %.4f\n", s, sim,
                    is_solved_and_feasible(am.model) ? objective_value(am.model) : NaN)
            flush(stdout)
        end
    end
    done = Set{Int}()
    isfile(out) && for l in readlines(out)[2:end]; push!(done, parse(Int, split(l, ",")[1])); end
    for s in first_scen:last_scen
        s in done && continue
        t = time(); am = build_anticipative_full(fd, s); tb = time() - t
        optimize!(am.model)
        @assert is_solved_and_feasible(am.model) "scenario $s: $(termination_status(am.model))"
        append_csv(out, "scenario,anticipative_cost,solve_time,mip_gap",
                   @sprintf("%d,%.4f,%.3f,%.2e", s, objective_value(am.model), solve_time(am.model), relative_gap(am.model)))
        @printf("scenario %4d  anticipative cost %.4f  (build %.1fs, solve %.1fs)\n", s, objective_value(am.model), tb, solve_time(am.model))
        flush(stdout)
    end
end
