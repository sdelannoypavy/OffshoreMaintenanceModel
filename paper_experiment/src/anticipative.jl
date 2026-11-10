# anticipative.jl
#
# Cost of the best anticipative policy of a scenario (Section "Setup", paragraph "Policy
# evaluation"): the degradations of all the components over the whole horizon are known in advance,
# and the maintenance is optimised accordingly by one MILP per scenario. Its cost is a lower bound on
# the cost of any policy on this scenario.
#
# Scenario s uses the same random draws as the simulation (scenario_draws, simulation.jl): component m,
# if it is not maintained in period t, moves at the end of the period with U[t, 1, m] and P_period,
# and it returns to the new state if it is maintained. The trajectory of a component is therefore a
# deterministic function of its maintenance periods, which the MILP represents by a path:
#   arc (m, i, j), i ≤ j, means "component m is new at the start of period i and next maintained in
#   period j", and its states between i and j are computed from the draws.
# The start days are on the grid FIRST_START_DAY, FIRST_START_DAY + STEP, … as in the simulations, so
# that a period is split into blocks of STEP days on which the capacity loss of a substation is constant.

using JuMP, Gurobi, Printf

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

"Number of slots of STEP days of the start-day grid in a period."
n_slots(fd::FluidData) = (fd.inst.T_period - FIRST_START_DAY + 1) ÷ STEP

# ─────────────────────────────────────────────────────────────────────────────
# Simplified model
# ─────────────────────────────────────────────────────────────────────────────

# The cost of a period is Σ_m T π ℓ_m(x_m) + Σ_{m maintained} c^rep_m + (number of maintenance days of
# every substation). The arcs carry the failure and replacement costs of their component, and the
# maintenance days are counted slot by slot. Only the costs of the arcs depend on the scenario, so
# that the MILP is built once and its objective is updated for every scenario.

struct AnticipativeSimplified
    model::Model
    arc::Dict{NTuple{3,Int},VariableRef}       # (m, i, j): new at period i, maintained in period j
    endarc::Dict{Tuple{Int,Int},VariableRef}   # (m, i): new at period i, never maintained again
    T::Int
end

function build_anticipative_simplified(fd::FluidData)
    inst = fd.inst; T = inst.n_years * N_PERIODS_PER_YEAR; nslots = n_slots(fd)
    π = constant_daily_penalty(fd)
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    arc = Dict{NTuple{3,Int},VariableRef}(); endarc = Dict{Tuple{Int,Int},VariableRef}()
    for m in 1:fd.M
        for i in 1:T
            for j in i:T
                arc[(m, i, j)] = @variable(model, binary = true)
            end
            endarc[(m, i)] = @variable(model, binary = true)
        end
        # path: node i is entered at period 1 (i = 1) or after a maintenance in period i − 1
        for i in 1:T
            inflow = i == 1 ? 1.0 : sum(arc[(m, ip, i - 1)] for ip in 1:(i - 1))
            @constraint(model, sum(arc[(m, i, j)] for j in i:T) + endarc[(m, i)] == inflow)
        end
    end
    obj = AffExpr(0.0)
    for t in 1:T
        cover = Dict{Tuple{Int,Int},AffExpr}()   # (m, slot) -> m under repair during the slot
        for m in 1:fd.M
            nd = fd.d[m] ÷ STEP
            slots = 1:(nslots - nd + 1)
            start = [@variable(model, binary = true) for k in slots]
            for k in slots, kk in k:(k + nd - 1)
                add_to_expression!(get!(() -> AffExpr(0.0), cover, (m, kk)), start[k])
            end
            @constraint(model, sum(start) == sum(arc[(m, i, t)] for i in 1:t))   # maintained in period t
        end
        for kk in 1:nslots
            @constraint(model, sum(fd.rho[m] * get(cover, (m, kk), AffExpr(0.0)) for m in 1:fd.M) <= inst.R)
            for p in 1:inst.n_poles   # u = 1 iff substation p is under maintenance during the slot
                u = @variable(model, lower_bound = 0, upper_bound = 1)
                for m in fd.comps_of_pole[p]
                    haskey(cover, (m, kk)) && @constraint(model, u >= cover[(m, kk)])
                end
                add_to_expression!(obj, π * STEP, u)
            end
        end
    end
    @objective(model, Min, obj)
    return AnticipativeSimplified(model, arc, endarc, T)
end

"Set the costs of the arcs of scenario `scen` and solve. Returns (cost, solve time, relative MIP gap)."
function solve_anticipative!(am::AnticipativeSimplified, fd::FluidData, scen::Int)
    U = scenario_draws(fd, scen)
    for m in 1:fd.M
        s = arc_states(fd, U, m, am.T)
        for i in 1:am.T
            cum = 0.0   # failure costs from period i to period j
            for j in i:am.T
                cum += failure_cost(fd, m, s[i, j])
                set_objective_coefficient(am.model, am.arc[(m, i, j)], cum + fd.cc[m])
            end
            set_objective_coefficient(am.model, am.endarc[(m, i)], cum)
        end
    end
    optimize!(am.model)
    @assert termination_status(am.model) == MOI.OPTIMAL "scenario $scen: $(termination_status(am.model))"
    return objective_value(am.model), solve_time(am.model), relative_gap(am.model)
end

# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

# The cost of a period is the true cost (eq:intro-cost), with the maximum of the capacity losses of a
# substation, the declared days and the annual quota. y[m, t, i] ∈ [0, 1] is 1 iff component m is new
# at period i and not maintained in i..t−1, so that its state in period t is s_m(i, t). On every block
# of days, the capacity loss of a substation is 1 if one of its components is under repair, and
# otherwise the largest loss of its components whose repair is not complete. f_b declared days are
# chosen in block b, Σ_b f_b = κ_t = STEP q_t, and the penalty of the block is π (|b| ℓ_b − f_b ℓ_b), the
# product f_b ℓ_b being written with the binary expansion of f_b (exact).

"Blocks of days of a period: (first day, last day, slot of the start-day grid or 0 if no repair can take place)."
function day_blocks(fd::FluidData)
    Tp = fd.inst.T_period; F = FIRST_START_DAY; nslots = n_slots(fd)
    blocks = Tuple{Int,Int,Int}[]
    F > 1 && push!(blocks, (1, F - 1, 0))
    for k in 1:nslots
        push!(blocks, (F + (k - 1) * STEP, F + k * STEP - 1, k))
    end
    F + nslots * STEP <= Tp && push!(blocks, (F + nslots * STEP, Tp, 0))
    return blocks
end

function build_anticipative_full(fd::FluidData, scen::Int)
    inst = fd.inst; T = inst.n_years * N_PERIODS_PER_YEAR; M = fd.M; nslots = n_slots(fd)
    π = constant_daily_penalty(fd)
    blocks = day_blocks(fd)
    U = scenario_draws(fd, scen)
    S = [arc_states(fd, U, m, T) for m in 1:M]
    ℓ(m, n) = capacity_loss(fd, m, n)

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(model)
    set_optimizer_attribute(model, "MIPGap", 1e-6)
    arc = Dict{NTuple{3,Int},VariableRef}(); y = Dict{NTuple{3,Int},VariableRef}()
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
    q = [@variable(model, integer = true, lower_bound = 0, upper_bound = fd.K0[p] ÷ STEP) for p in 1:inst.n_poles, t in 1:T]
    for t in 1:T
        # maintenance decisions and start slots
        maint = Dict{Int,AffExpr}(); cover = Dict{Tuple{Int,Int},AffExpr}(); done = Dict{Tuple{Int,Int},AffExpr}()
        for m in 1:M
            nd = fd.d[m] ÷ STEP
            slots = 1:(nslots - nd + 1)
            start = Dict(k => @variable(model, binary = true) for k in slots)
            for k in slots, kk in k:(k + nd - 1)
                add_to_expression!(get!(() -> AffExpr(0.0), cover, (m, kk)), start[k])
            end
            maint[m] = sum(start[k] for k in slots)
            @constraint(model, maint[m] == sum(arc[(m, i, t)] for i in 1:t))
            add_to_expression!(obj, fd.cc[m], maint[m])
            for (bi, (_, _, k)) in enumerate(blocks)
                # repair of m complete before block bi: started at a slot that ends before it
                done[(m, bi)] = k == 0 ? (bi == 1 ? AffExpr(0.0) : 1.0 * maint[m]) :
                                sum((start[kk] for kk in slots if kk + nd - 1 < k); init = AffExpr(0.0))
            end
        end
        for k in 1:nslots   # technician capacity
            @constraint(model, sum(fd.rho[m] * get(cover, (m, k), AffExpr(0.0)) for m in 1:M) <= inst.R)
        end
        # capacity loss, declared days and penalty of every substation and block
        for p in 1:inst.n_poles
            ms = fd.comps_of_pole[p]
            Lm = Dict(m => sum((ℓ(m, S[m][i, t]) * y[(m, t, i)] for i in 1:t if ℓ(m, S[m][i, t]) > 0); init = AffExpr(0.0)) for m in ms)
            declared = AffExpr(0.0)
            for (bi, (d1, d2, k)) in enumerate(blocks)
                len = d2 - d1 + 1
                loss = @variable(model, lower_bound = 0, upper_bound = 1)
                if k > 0   # maintenance day of the substation
                    for m in ms
                        haskey(cover, (m, k)) && @constraint(model, loss >= cover[(m, k)])
                    end
                end
                for m in ms   # loss of the components whose repair is not complete
                    isempty(Lm[m].terms) || @constraint(model, loss >= Lm[m] - done[(m, bi)])
                end
                # f declared days in the block, binary expansion, z_i = bit_i × loss
                nb = floor(Int, log2(len)) + 1
                bits = [@variable(model, binary = true) for _ in 1:nb]
                f = sum(2^(i - 1) * bits[i] for i in 1:nb)
                @constraint(model, f <= len)
                z = [@variable(model, lower_bound = 0) for _ in 1:nb]
                for i in 1:nb
                    @constraint(model, z[i] <= bits[i]); @constraint(model, z[i] <= loss)
                end
                add_to_expression!(obj, π * len, loss)
                add_to_expression!(obj, -π, sum(2^(i - 1) * z[i] for i in 1:nb))
                add_to_expression!(declared, f)
            end
            @constraint(model, declared == STEP * q[p, t])   # κ_t of the substation
        end
    end
    for p in 1:inst.n_poles, year in 1:inst.n_years   # annual quota of every substation
        ts = ((year - 1) * N_PERIODS_PER_YEAR + 1):(year * N_PERIODS_PER_YEAR)
        @constraint(model, STEP * sum(q[p, t] for t in ts) <= fd.K0[p])
    end
    @objective(model, Min, obj)
    return model
end

# ─────────────────────────────────────────────────────────────────────────────
# Both models
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_anticipative_solver(fd)

Function scen -> (anticipative cost, solve time, relative MIP gap) of scenario `scen`, for the model of
the instance (the MILP of the simplified model is built once).
"""
function make_anticipative_solver(fd::FluidData)
    if fd.inst.simplified
        am = build_anticipative_simplified(fd)
        return scen -> solve_anticipative!(am, fd, scen)
    end
    return function (scen)
        model = build_anticipative_full(fd, scen)
        optimize!(model)
        @assert is_solved_and_feasible(model) "scenario $scen: $(termination_status(model))"
        return objective_value(model), solve_time(model), relative_gap(model)
    end
end
