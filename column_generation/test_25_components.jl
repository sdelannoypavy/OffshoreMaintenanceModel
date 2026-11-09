# =============================================================================
# 25-component instance: comparison of two solution methods
#
#   Method 1 (:cg)    : fluid LP + column generation
#                        (RMP + pricing MILP, see test_25_composants.tex)
#   Method 2 (:relax) : relaxed LP solved directly, then rounded (nearest /
#                        argmax), with a repair heuristic
#                        (see the "Repair heuristic" section of the .tex)
#
# Mirrors the structure of script_simple_pole.jl: at each stage (60 days),
# solve an LP (rolling fluid horizon H), actually simulate the stage,
# compute the cost, write to files, and repeat. Policies already computed
# for a previously-encountered state are cached and read back from the
# corresponding CSV file (as in collect_policies).
# =============================================================================

using JuMP, Gurobi, CSV, DataFrames, Printf, Random, Statistics, Distributions, Plots
gr()

const N_SIM = 5

# tolerance of column generation (pricing MILP) for adding a new column to the RMP.
# Widened from 1e-1: reduced costs here are typically in the thousands, so requiring
# convergence down to -0.1 forces Gurobi to nail a very tight, hard-to-locate threshold
# (via BestObjStop) even when the true optimum is barely negative. -10 is still tight
# relative to the typical -1000..-9000 range of a genuinely improving column, but lets
# BestObjStop trigger as soon as a "clearly good enough" solution is found, instead of
# requiring near-exact convergence to the true pricing optimum.
const tol = 10.0

const max_iter::Int = 3 #nb max d'itérations de la génération de colonnes (needs `const`:
# a non-const global re-read every loop iteration prevents Julia from specializing on its
# type, which is measurably slower for a value read in a hot loop like this one)

const VERBOSE = true #print progress logs (policy solves, CG iterations, per-stage progress)

# @logmsg(fmt, args...) prints only when VERBOSE is true; used for all progress logging
# below (kept separate from CSV/PDF output, which is always written regardless of VERBOSE).
# Implemented as a macro (not a function) because @printf requires a literal format string.
macro logmsg(fmt, args...)
    return :(VERBOSE && @printf($fmt, $(map(esc, args)...)))
end

const SCHEDULE_PLOT_DIR = "schedules_25_components"

# -----------------------------------------------------------------------------
# 1. Instance
# -----------------------------------------------------------------------------

const C          = 25
const MTBF       = 40.0
const K0         = 0      # annual quota of free maintenance days
const R          = 3       # technicians
const DAY        = 60      # days per stage
const STAGE_YEAR = 6       # stages per year
const YEARS      = 20
const TBAR       = YEARS * STAGE_YEAR   # 120 stages
const REPL_COST  = 100.0
const PENALTY    = 100.0
const H          = 6       # rolling fluid horizon (1 year), as in script_simple_pole.jl

Random.seed!(2026)
const d = rand(1:10, C)     # maintenance duration (fixed instance data)

const p_day = 2 / (MTBF * 365)
const P1 = [1-p_day 0.0 0.0; p_day 1-p_day 0.0; 0.0 p_day 1.0]
const P_period = P1^DAY     # same 60-day transition matrix for every component

# Ptrans(x_prev, x_next, m): m=1 -> no maintenance (P_period), m=2 -> certain reset to state 1
Ptrans(xp::Int, xn::Int, m::Int) = m == 2 ? (xn == 1 ? 1.0 : 0.0) : P_period[xn, xp]

# Failure cost with no maintenance and no free days (see "Precomputed costs" section of
# the .tex), for a component starting the stage in state x0. Unlike a flat "PENALTY*DAY
# iff x0==3" rule, this must account for the non-zero probability that a component
# starting in state 1 or 2 reaches state 3 partway through the 60-day stage. As in the
# compute_δ_simple / average_period_cost logic of column_generation.jl, this is computed
# by propagating the state probability distribution forward day-by-day using the DAILY
# transition matrix P1 (not the aggregated 60-day P_period), and summing the expected
# penalty (PENALTY * P(state=3)) over all 60 days.
function compute_delta(x0::Int)
    state = zeros(3)
    state[x0] = 1.0
    total = PENALTY * state[3]          # day 1: state as observed at the start of the stage
    for _ in 2:DAY
        state = P1 * state              # one more day of degradation
        total += PENALTY * state[3]
    end
    return total
end

const DELTA_TABLE = [compute_delta(x) for x in 1:3]
delta(x::Int) = DELTA_TABLE[x]

const GRB_ENV = Gurobi.Env()
optimizer_fn() = Gurobi.Optimizer(GRB_ENV)

# -----------------------------------------------------------------------------
# 2. Exact simulation of a stage (true joint cost, not the δ/η approximation)
# -----------------------------------------------------------------------------

"""
    simulate_stage(x0, comps, start_days, free_days)

Simulates the 60 days of a stage starting from state x0 (Vector{Int}, length C),
given the plan (comps, start_days, free_days). Returns (cost, x_new).

Failures are drawn day by day using the daily transition matrix P1 (not a single
draw from the aggregated P_period): for each component not currently under
maintenance, its state is sampled one day at a time via `rand(Categorical(P1[:,
cur]))`, so a component starting the stage in state 1 or 2 can genuinely reach
state 3 partway through the 60 days (and, symmetrically, a component whose
maintenance completes early in the stage can keep degrading afterwards). On each
day τ, a penalty is paid iff τ is not a free day AND (at least one component is
under maintenance that day OR at least one component is in state 3 that day).
"""
function simulate_stage(x0::Vector{Int}, comps::Vector{Int},
                         start_days::Dict{Int,Int}, free_days::Vector{Int})
    active = falses(DAY, C)
    for c in comps
        s = start_days[c]
        for τ in s:min(s + d[c] - 1, DAY)
            active[τ, c] = true
        end
    end
    is_free = falses(DAY)
    for τ in free_days
        is_free[τ] = true
    end

    failed = falses(DAY, C)
    x_new = copy(x0)

    for c in 1:C
        cur = x0[c]
        for τ in 1:DAY
            if active[τ, c]
                # under maintenance this day: no failure tracking, reset at completion
                if τ == start_days[c] + d[c] - 1
                    cur = 1
                end
                continue
            end
            if cur == 3
                failed[τ, c] = true
            end
            cur = rand(Distributions.Categorical(P1[:, cur]))
        end
        x_new[c] = cur
    end

    downtime = 0.0
    for τ in 1:DAY
        down = any(@view active[τ, :]) || any(@view failed[τ, :])
        if down && !is_free[τ]
            downtime += PENALTY
        end
    end
    repl = REPL_COST * length(comps)

    return downtime + repl, x_new
end

# -----------------------------------------------------------------------------
# 3. Repair heuristic (technician capacity)
#    -> described in test_25_composants.tex, "Repair heuristic" section
# -----------------------------------------------------------------------------

"""
    repair_schedule(candidates, desired_start, confidence)

Repairs a candidate schedule so that it respects the capacity of R parallel
technicians. Components are processed in order (increasing desired start day,
ties broken by decreasing confidence) and each is placed on the earliest
feasible day >= its desired start day; if no day works within the window
[desired start day, DAY-d[c]+1], earlier days are also tried; if still no day
works, the component is deferred (dropped from this stage's plan).
"""
function repair_schedule(candidates::Vector{Int}, desired_start::Dict{Int,Int},
                          confidence::Dict{Int,Float64})
    order = sort(candidates, by = c -> (desired_start[c], -confidence[c]))
    occ = zeros(Int, DAY)
    final_start = Dict{Int,Int}()
    kept = Int[]
    for c in order
        placed = false
        for s in desired_start[c]:(DAY - d[c] + 1)
            if all(occ[τ] < R for τ in s:(s + d[c] - 1))
                for τ in s:(s + d[c] - 1); occ[τ] += 1; end
                final_start[c] = s; push!(kept, c); placed = true; break
            end
        end
        if !placed
            for s in 1:(desired_start[c] - 1)
                if all(occ[τ] < R for τ in s:(s + d[c] - 1))
                    for τ in s:(s + d[c] - 1); occ[τ] += 1; end
                    final_start[c] = s; push!(kept, c); placed = true; break
                end
            end
        end
        # otherwise: c is deferred (not maintained this stage)
    end
    return kept, final_start
end

# -----------------------------------------------------------------------------
# 3b. Gantt-style schedule plot, saved as PDF (mirrors column_generation/plot_action.jl)
# -----------------------------------------------------------------------------

"""
    plot_and_save_schedule(comps, start_days, free_days, method_id, season, stage_tag)

Renders a Gantt-style chart of one stage's maintenance schedule and saves it
as a PDF in `SCHEDULE_PLOT_DIR`. The filename encodes the solution method
(1 = column generation, 2 = relaxation + rounding + repair) and the season
(local stage number modulo 6, i.e. t0_local), as requested, plus the global
stage number `stage_tag` to keep filenames unique across different states
sharing the same (method, season).

Layout: y-axis = components 1..C, x-axis = calendar day within the 60-day
stage, yellow bands = free (penalty-exempt) days, coloured bars = maintenance
operations. Same visual style as column_generation/plot_action.jl.
"""
function plot_and_save_schedule(comps::Vector{Int}, start_days::Dict{Int,Int},
                                 free_days::Vector{Int}, method_id::Int, season::Int,
                                 stage_tag::String)
    mkpath(SCHEDULE_PLOT_DIR)

    y_labels = ["Comp $c" for c in 1:C]
    plot_height = max(380, 18 * C)

    plt = plot(
        xlims         = (0.5, DAY + 0.5),
        ylims         = (0.5, C + 0.5),
        yticks        = (1:C, y_labels),
        xlabel        = "Day of stage",
        title         = "Method $method_id — season $season",
        legend        = :outertop,
        legendcolumns = -1,
        size          = (1000, plot_height),
        grid          = :x,
        framestyle    = :box,
        left_margin   = 12Plots.mm,
        bottom_margin = 8Plots.mm,
        top_margin    = 8Plots.mm,
        right_margin  = 8Plots.mm,
    )

    first_free = true
    for τ in free_days
        plot!(plt,
            Shape([τ - 0.5, τ + 0.5, τ + 0.5, τ - 0.5], [0.5, 0.5, C + 0.5, C + 0.5]),
            fillcolor = :gold, linecolor = :transparent, alpha = 0.35,
            label = first_free ? "Free days" : "")
        first_free = false
    end

    palette = [:steelblue, :darkorange, :green3, :crimson, :mediumpurple, :saddlebrown, :teal,
               :dodgerblue, :orchid, :goldenrod]
    for c in comps
        s = start_days[c]
        e = s + d[c] - 1
        bar_c = palette[mod1(c, length(palette))]
        plot!(plt,
            Shape([s - 0.42, e + 0.42, e + 0.42, s - 0.42], [c - 0.33, c - 0.33, c + 0.33, c + 0.33]),
            fillcolor = bar_c, linecolor = bar_c, alpha = 0.80, label = "")
    end

    major_ticks = sort(unique(vcat(1, collect(5:5:DAY), DAY)))
    plot!(plt, xticks = major_ticks)

    pdf_path = joinpath(SCHEDULE_PLOT_DIR, "schedule_method$(method_id)_season$(season)_$(stage_tag).pdf")
    savefig(plt, pdf_path)
    @logmsg("  [plot] Saved: %s\n", pdf_path)
    return pdf_path
end

# -----------------------------------------------------------------------------
# 4. Pricing MILP (Method 1)
#    ong[c,τ] is written directly as a linear expression of mstart (no q^v
#    counter nor integer "ong" variable): without weather, an operation
#    always runs for exactly d_c consecutive days without interruption, so
#    ong[c,τ] = 1 iff a start happened within the last d_c days -- which is
#    automatically binary since sum_τ mstart[c,τ] = z[c] <= 1 (at most one
#    active start day).
#
#    `am` (at least one component under maintenance) and `w` (McCormick
#    linearisation of am ∧ f) are merged into a single variable `pay[τ]`,
#    equal to 1 iff some component is under maintenance (ong=1) AND the day
#    is not free (f=0): pay[τ] >= ong[c,τ] - f[τ] for every c. Since ong and f
#    are binary, max_c(ong[c,τ] - f[τ]) ∈ {-1,0,1}, and minimising pay[τ]
#    forces it to exactly that value at the optimum -- so pay does not need
#    to be declared binary, it is automatically integral.
# -----------------------------------------------------------------------------

function solve_pricing(lambda::Matrix{Float64}, k_avail::Int; time_limit::Float64=30.0, mip_gap::Float64=0.10,
                        warm_start=nothing)
    # Column generation only needs *a* column with reduced cost < -tol, not the exactly
    # optimal (most negative) one -- most of a MILP's solve time is typically spent
    # improving/proving the LAST bit of the objective, not finding a first good solution.
    # BestObjStop tells Gurobi to stop the instant it finds a feasible solution at or
    # below -tol, which is exactly our stopping criterion -- this is normally what ends
    # the solve (fast, once any improving column is found). MIPGap and TimeLimit remain
    # as fallbacks for the case where no improving column exists at all (redcost >= -tol
    # is the true optimum), so Gurobi still needs to search/prove that within a bounded time.
    model = Model(optimizer_fn)
    set_silent(model)
    set_optimizer_attribute(model, "OutputFlag",   0)
    set_optimizer_attribute(model, "TimeLimit",    time_limit)
    set_optimizer_attribute(model, "MIPGap",       mip_gap)
    set_optimizer_attribute(model, "BestObjStop",  -tol)

    @variable(model, z[1:C], Bin)
    @variable(model, mstart[c=1:C, τ=1:(DAY - d[c] + 1)], Bin)
    @variable(model, f[1:DAY], Bin)
    @variable(model, 0 <= pay[1:DAY] <= 1)
    @variable(model, ksel[0:k_avail], Bin)
    @variable(model, y[1:C, 0:k_avail], Bin)

    @constraint(model, [c=1:C], sum(mstart[c, τ] for τ in 1:(DAY - d[c] + 1)) == z[c])

    ong = Matrix{AffExpr}(undef, C, DAY)
    for c in 1:C, τ in 1:DAY
        lo = max(1, τ - d[c] + 1)
        hi = min(τ, DAY - d[c] + 1)
        ong[c, τ] = hi >= lo ? sum(mstart[c, τ2] for τ2 in lo:hi) : AffExpr(0.0)
    end

    @constraint(model, [τ=1:DAY], sum(ong[c, τ] for c in 1:C) <= R)
    @constraint(model, [c=1:C, τ=1:DAY], pay[τ] >= ong[c, τ] - f[τ])
    @constraint(model, sum(ksel[κ] for κ in 0:k_avail) == 1)
    @constraint(model, sum(f[τ] for τ in 1:DAY) == sum(κ * ksel[κ] for κ in 0:k_avail))
    @constraint(model, [c=1:C, κ=0:k_avail], y[c, κ] <= z[c])
    @constraint(model, [c=1:C, κ=0:k_avail], y[c, κ] <= ksel[κ])
    @constraint(model, [c=1:C, κ=0:k_avail], y[c, κ] >= z[c] + ksel[κ] - 1)

    @objective(model, Min,
        PENALTY * sum(pay[τ] for τ in 1:DAY) +
        sum(lambda[c, κ+1] * y[c, κ] for c in 1:C, κ in 0:k_avail))

    # Warm start from the previous iteration's pricing solution for this same h. Duals
    # only shift a little between consecutive CG iterations, so the previous near-optimal
    # (comps, start_days, free_days) is usually still a very good (or the) solution here --
    # giving Gurobi this as an MIP start lets it verify/refine instead of searching from
    # scratch, which is where most of the solve time was going.
    if warm_start !== nothing
        κ_ws = min(length(warm_start.free_days), k_avail)
        for c in 1:C
            in_ws = c in warm_start.comps
            set_start_value(z[c], in_ws ? 1.0 : 0.0)
            for τ in 1:(DAY - d[c] + 1)
                s_ws = in_ws ? warm_start.start_days[c] : 0
                set_start_value(mstart[c, τ], (τ == s_ws) ? 1.0 : 0.0)
            end
            for κ in 0:k_avail
                set_start_value(y[c, κ], (in_ws && κ == κ_ws) ? 1.0 : 0.0)
            end
        end
        for τ in 1:DAY
            set_start_value(f[τ], (τ in warm_start.free_days) ? 1.0 : 0.0)
        end
        for κ in 0:k_avail
            set_start_value(ksel[κ], κ == κ_ws ? 1.0 : 0.0)
        end
    end

    optimize!(model)

    zc = value.(z)
    comps = [c for c in 1:C if zc[c] > 0.5]
    start_days = Dict(c => (findfirst(τ -> value(mstart[c, τ]) > 0.5, 1:(DAY - d[c] + 1)))
                       for c in comps)
    fτ = value.(f)
    free_days = [τ for τ in 1:DAY if fτ[τ] > 0.5]

    return (comps=comps, start_days=start_days, free_days=free_days), objective_value(model)
end

empty_column() = (comps=Int[], start_days=Dict{Int,Int}(), free_days=Int[])

function stage_eta(b)
    active = falses(DAY)
    for c in b.comps
        s = b.start_days[c]
        for τ in s:min(s + d[c] - 1, DAY)
            active[τ] = true
        end
    end
    is_free = falses(DAY)
    for τ in b.free_days
        is_free[τ] = true
    end
    return PENALTY * sum((active[τ] && !is_free[τ]) ? 1.0 : 0.0 for τ in 1:DAY)
end

# -----------------------------------------------------------------------------
# 5. RMP (restricted master problem), rolling fluid horizon H
#    Practical simplification: beyond the 1st local stage (h>1), the pricing
#    step assumes the full annual quota K0 is available (rather than tracking
#    its expected depletion exactly), since only the h=1 decision is ever
#    actually applied -- see the remark in the .tex.
#
#    IMPORTANT: the base model (q variables, flow/init constraints, coupling
#    and convexity constraint *shells*) is built exactly ONCE per call to
#    solve_RMP_CG, not once per column-generation iteration. Only the columns
#    themselves are added incrementally (add_column!), mirroring add_column!
#    in column_generation.jl. Rebuilding the whole model (108,900 q variables,
#    tens of thousands of constraints) from scratch at every iteration was the
#    actual cause of multi-minute iterations -- Gurobi itself solves an LP this
#    size quickly; it was the repeated Julia/JuMP model construction that
#    dominated the runtime.
# -----------------------------------------------------------------------------

function build_RMP_base(x0::Vector{Int}, k0::Int, t0_global::Int)
    model = Model(optimizer_fn)
    set_silent(model)

    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "Method", 1)

    @variable(model, q[1:C, 1:3, 1:2, 0:K0, 0:K0, 1:H] >= 0)
    @constraint(model, [c=1:C, x=1:3, m=1:2, k=0:K0-1, κ=(k+1):K0, h=1:H], q[c, x, m, k, κ, h] == 0)

    @constraint(model, [c=1:C, x=1:3],
        sum(q[c, x, m, k0, κ, 1] for m in 1:2, κ in 0:k0) == (x0[c] == x ? 1.0 : 0.0))
    for k in 0:K0
        if k != k0
            @constraint(model, [c=1:C, x=1:3], sum(q[c, x, m, k, κ, 1] for m in 1:2, κ in 0:k) == 0.0)
        end
    end

    for h in 2:H
        gt = t0_global + h - 1
        if gt % STAGE_YEAR == 1
            @constraint(model, [c=1:C, x=1:3],
                sum(q[c, x, m, K0, κ, h] for m in 1:2, κ in 0:K0) >=
                sum(Ptrans(xp, x, m) * q[c, xp, m, k, κ, h-1] for xp in 1:3, m in 1:2, k in 0:K0 for κ in 0:k))
            @constraint(model, [c=1:C, x=1:3, k=0:K0-1],
                sum(q[c, x, m, k, κ, h] for m in 1:2, κ in 0:k) == 0.0)
        else
            for k in 0:K0
                @constraint(model, [c=1:C, x=1:3],
                    sum(q[c, x, m, k, κ, h] for m in 1:2, κ in 0:k) >=
                    sum(Ptrans(xp, x, m) * q[c, xp, m, k+κp, κp, h-1]
                        for xp in 1:3, m in 1:2, κp in 0:(K0-k)))
            end
        end
    end

    # Coupling constraint shells (RHS = 0): each new column contributes a term
    # via add_column!'s set_normalized_coefficient call, exactly as maint_prob
    # is populated in column_generation.jl.
    coupling = Dict{Tuple{Int,Int,Int}, ConstraintRef}()
    for h in 1:H, c in 1:C, κ in 0:K0
        coupling[(c, κ, h)] = @constraint(model, sum(q[c, x, 2, k, κ, h] for x in 1:3, k in κ:K0) <= 0)
    end

    # Convexity constraint shells (sum_i B[h,i] == 1): built empty (0 == 1),
    # populated incrementally as columns are added.
    convexity = Dict{Int, ConstraintRef}()
    for h in 1:H
        convexity[h] = @constraint(model, 0 == 1)
    end

    δobj  = sum(delta(x) * q[c, x, 1, k, κ, h] for c in 1:C, x in 1:3, k in 0:K0 for κ in 0:k for h in 1:H)
    replo = sum(REPL_COST * q[c, x, 2, k, κ, h] for c in 1:C, x in 1:3, k in 0:K0 for κ in 0:k for h in 1:H)
    @objective(model, Min, δobj + replo)   # η contribution of each column added via set_objective_coefficient

    return model, coupling, convexity
end

"""
    add_column!(model, coupling, convexity, col, h)

Adds one new column (joint action `col`) to stage `h` of an already-built RMP,
without rebuilding the model: one new variable, its objective coefficient
(η(col)), its coefficient in the convexity constraint for stage h, and its
coefficient (-1) in the coupling constraint for each maintained component at
col's κ. Returns the new variable.
"""
function add_column!(model, coupling::Dict, convexity::Dict, col, h::Int)
    v = @variable(model, lower_bound = 0)
    set_objective_coefficient(model, v, stage_eta(col))
    set_normalized_coefficient(convexity[h], v, 1.0)
    κ_col = length(col.free_days)
    for c in col.comps
        set_normalized_coefficient(coupling[(c, κ_col, h)], v, -1.0)
    end
    return v
end

function solve_RMP_CG(x0::Vector{Int}, k0::Int, t0_global::Int)
    t_start = time()
    @logmsg("    [CG] solving new policy (x0=%s, k0=%d, t0=%d)...\n", string(x0), k0, t0_global)

    build_t = time()
    model, coupling, convexity = build_RMP_base(x0, k0, t0_global)
    @logmsg("    [CG]   base model built in %.1fs\n", time() - build_t)

    Bpool = Vector{Any}[[empty_column()] for _ in 1:H]
    Bvars = [VariableRef[] for _ in 1:H]
    for h in 1:H
        push!(Bvars[h], add_column!(model, coupling, convexity, Bpool[h][1], h))
    end

    # Warm start for each h's pricing MILP: the solution found last time pricing was
    # solved for that h (whether or not it ended up being added to Bpool), since duals
    # only shift a little between consecutive CG iterations.
    last_sol = fill(nothing, H)

    for iter in 1:max_iter
        iter_t = time()
        rmp_t = time()
        optimize!(model)
        rmp_elapsed = time() - rmp_t
        @logmsg("    [CG]   iter %d: RMP solve %.1fs (status %s)\n",
                iter, rmp_elapsed, termination_status(model))

        # Read ALL duals and solve ALL pricing MILPs first, without touching the RMP
        # model -- adding a variable to `model` invalidates the just-computed solution
        # (JuMP requires a fresh optimize! before the next dual() query), so columns
        # must only be added to the model AFTER every h has been priced this iteration.
        pending = Tuple{Int,Any}[]
        pricing_elapsed = 0.0
        for h in 1:H
            k_avail = h == 1 ? k0 : K0
            lambda = zeros(C, k_avail + 1)
            for c in 1:C, κ in 0:k_avail
                lambda[c, κ+1] = dual(coupling[(c, κ, h)])
            end
            h_t = time()
            col, redcost = solve_pricing(lambda, k_avail; warm_start=last_sol[h])
            last_sol[h] = col
            h_elapsed = time() - h_t
            pricing_elapsed += h_elapsed
            @logmsg("    [CG]     h=%d pricing %.1fs (redcost=%.3f)\n", h, h_elapsed, redcost)
            if redcost < -tol && !(col in Bpool[h])
                push!(Bpool[h], col)
                push!(pending, (h, col))
            end
        end

        for (h, col) in pending
            push!(Bvars[h], add_column!(model, coupling, convexity, col, h))
        end
        improved = !isempty(pending)
        @logmsg("    [CG]   iteration %d done in %.1fs total (RMP %.1fs + pricing %.1fs) (%s new column%s)\n",
               iter, time() - iter_t, rmp_elapsed, pricing_elapsed, improved ? "some" : "no", improved ? "s" : "")
        improved || break
    end

    optimize!(model)
    Bvals1 = [value(v) for v in Bvars[1]]
    i_star = argmax(Bvals1)
    chosen = Bpool[1][i_star]
    elapsed = time() - t_start
    @logmsg("    [CG] policy found in %.1fs total\n", elapsed)
    return chosen.comps, chosen.start_days, chosen.free_days, elapsed
end

# -----------------------------------------------------------------------------
# 6. Relaxed LP (Method 2), then rounding + repair
# -----------------------------------------------------------------------------

function build_relaxed_LP(x0::Vector{Int}, k0::Int, t0_global::Int)
    model = Model(optimizer_fn)
    set_silent(model)

    @variable(model, q[1:C, 1:3, 1:2, 0:K0, 0:K0, 1:H] >= 0)
    @constraint(model, [c=1:C, x=1:3, m=1:2, k=0:K0-1, κ=(k+1):K0, h=1:H], q[c, x, m, k, κ, h] == 0)

    @constraint(model, [c=1:C, x=1:3],
        sum(q[c, x, m, k0, κ, 1] for m in 1:2, κ in 0:k0) == (x0[c] == x ? 1.0 : 0.0))
    for k in 0:K0
        if k != k0
            @constraint(model, [c=1:C, x=1:3], sum(q[c, x, m, k, κ, 1] for m in 1:2, κ in 0:k) == 0.0)
        end
    end

    for h in 2:H
        gt = t0_global + h - 1
        if gt % STAGE_YEAR == 1
            @constraint(model, [c=1:C, x=1:3],
                sum(q[c, x, m, K0, κ, h] for m in 1:2, κ in 0:K0) >=
                sum(Ptrans(xp, x, m) * q[c, xp, m, k, κ, h-1] for xp in 1:3, m in 1:2, k in 0:K0 for κ in 0:k))
            @constraint(model, [c=1:C, x=1:3, k=0:K0-1],
                sum(q[c, x, m, k, κ, h] for m in 1:2, κ in 0:k) == 0.0)
        else
            for k in 0:K0
                @constraint(model, [c=1:C, x=1:3],
                    sum(q[c, x, m, k, κ, h] for m in 1:2, κ in 0:k) >=
                    sum(Ptrans(xp, x, m) * q[c, xp, m, k+κp, κp, h-1]
                        for xp in 1:3, m in 1:2, κp in 0:(K0-k)))
            end
        end
    end

    @variable(model, 0 <= z[1:C, 1:H] <= 1)
    @variable(model, 0 <= mstart[c=1:C, τ=1:(DAY - d[c] + 1), h=1:H] <= 1)
    @variable(model, 0 <= f[1:DAY, 1:H] <= 1)
    @variable(model, 0 <= pay[1:DAY, 1:H] <= 1)

    @constraint(model, [c=1:C, h=1:H], sum(mstart[c, τ, h] for τ in 1:(DAY - d[c] + 1)) == z[c, h])

    ong = Array{AffExpr}(undef, C, DAY, H)
    for c in 1:C, τ in 1:DAY, h in 1:H
        lo = max(1, τ - d[c] + 1)
        hi = min(τ, DAY - d[c] + 1)
        ong[c, τ, h] = hi >= lo ? sum(mstart[c, τ2, h] for τ2 in lo:hi) : AffExpr(0.0)
    end

    @constraint(model, [τ=1:DAY, h=1:H], sum(ong[c, τ, h] for c in 1:C) <= R)
    @constraint(model, [c=1:C, τ=1:DAY, h=1:H], pay[τ, h] >= ong[c, τ, h] - f[τ, h])

    @constraint(model, [c=1:C, κ=0:K0, h=1:H],
        sum(q[c, x, 2, k, κ, h] for x in 1:3, k in κ:K0) <= z[c, h])
    @constraint(model, [h=1:H],
        sum(κ * sum(q[1, x, m, k, κ, h] for x in 1:3, m in 1:2) for k in 0:K0 for κ in 0:k) ==
        sum(f[τ, h] for τ in 1:DAY))
    @constraint(model, [h=1:H], sum(f[τ, h] for τ in 1:DAY) <= K0)

    δobj  = sum(delta(x) * q[c, x, 1, k, κ, h] for c in 1:C, x in 1:3, k in 0:K0 for κ in 0:k for h in 1:H)
    replo = sum(REPL_COST * z[c, h] for c in 1:C, h in 1:H)
    ηobj  = sum(PENALTY * pay[τ, h] for τ in 1:DAY, h in 1:H)

    @objective(model, Min, δobj + replo + ηobj)
    return model, z, mstart, f
end

"""
Rounds the fractional solution of the relaxed LP at local stage h=1 and
repairs the resulting schedule (see the "Repair heuristic" section of the
.tex):
  1. z[c,1] rounded to the nearest integer (>=0.5 -> maintained);
  2. for each retained component, start day = argmax_τ mstart[c,τ,1];
  3. candidate free days = {τ : f[τ,1] >= 0.5}; if their number exceeds the
     quota k0 actually available, only the k0 days with the highest
     fractional value f[τ,1] are kept (repair of the quota constraint);
  4. the schedule of the retained components is then repaired by
     repair_schedule to respect the capacity of R parallel technicians
     (components that cannot be fitted within the 60-day window are
     deferred to the next stage).
"""
function solve_relaxed_and_round(x0::Vector{Int}, k0::Int, t0_global::Int)
    t_start = time()
    @logmsg("    [relax] solving new policy (x0=%s, k0=%d, t0=%d)...\n", string(x0), k0, t0_global)
    model, z, mstart, f = build_relaxed_LP(x0, k0, t0_global)
    optimize!(model)

    zc = value.(z)[:, 1]
    fτ = value.(f)[:, 1]

    z_round = [zc[c] >= 0.5 for c in 1:C]
    candidates = [c for c in 1:C if z_round[c]]

    desired_start = Dict{Int,Int}()
    confidence = Dict{Int,Float64}()
    for c in candidates
        τs = 1:(DAY - d[c] + 1)
        mvals = [value(mstart[c, τ, 1]) for τ in τs]
        desired_start[c] = collect(τs)[argmax(mvals)]
        confidence[c] = zc[c]
    end

    kept, final_start = repair_schedule(candidates, desired_start, confidence)

    free_candidates = [τ for τ in 1:DAY if fτ[τ] >= 0.5]
    if length(free_candidates) > k0
        sort!(free_candidates, by = τ -> -fτ[τ])
        free_candidates = free_candidates[1:k0]
    end

    elapsed = time() - t_start
    return kept, final_start, free_candidates, elapsed
end

# -----------------------------------------------------------------------------
# 7. Policy cache (as in collect_policies / the cache of script_simple_pole.jl)
# -----------------------------------------------------------------------------

const PolicyVal = Tuple{Vector{Int}, Dict{Int,Int}, Vector{Int}, Float64}

"""
    load_cache(file_name)

Loads a previously written policy cache. Each row's expected width depends on
the current C and DAY (3*C + 3 + DAY columns: x0, k0, t0, comps_ind, starts,
free_ind, solve_t). If the file was written by a run with different C/DAY
(e.g. after changing the instance size), its rows will have the wrong width;
rather than crashing with a BoundsError, this is detected up front and the
stale cache is discarded (with a warning), so the run starts fresh instead of
failing. Delete the file yourself if you want a clean run without the warning.
"""
function load_cache(file_name::String)
    cache = Dict{Tuple{Vector{Int},Int,Int}, PolicyVal}()
    expected_cols = 3 * C + 3 + DAY
    if isfile(file_name)
        df = CSV.read(file_name, DataFrame; header=false)
        if ncol(df) != expected_cols
            @warn "Ignoring stale cache $(file_name): it has $(ncol(df)) columns, " *
                  "expected $(expected_cols) for the current C=$(C), DAY=$(DAY) " *
                  "(the file was likely written with different instance parameters). " *
                  "Delete it manually to silence this warning."
            return cache
        end
        for row in eachrow(df)
            v = collect(row)
            x0 = Int.(v[1:C]); k0 = Int(v[C+1]); t0 = Int(v[C+2])
            comps_ind = Int.(v[(C+3):(2C+2)])
            starts    = Int.(v[(2C+3):(3C+2)])
            free_ind  = Int.(v[(3C+3):(3C+2+DAY)])
            solve_t   = Float64(v[3C+3+DAY])
            comps = [c for c in 1:C if comps_ind[c] == 1]
            sd = Dict(c => starts[c] for c in comps)
            fd = [τ for τ in 1:DAY if free_ind[τ] == 1]
            cache[(x0, k0, t0)] = (comps, sd, fd, solve_t)
        end
    end
    return cache
end

function append_policy_row(file_name::String, x0::Vector{Int}, k0::Int, t0::Int,
                            comps::Vector{Int}, start_days::Dict{Int,Int},
                            free_days::Vector{Int}, solve_t::Float64)
    comps_ind = [c in comps ? 1 : 0 for c in 1:C]
    starts    = [c in comps ? start_days[c] : 0 for c in 1:C]
    free_ind  = [τ in free_days ? 1 : 0 for τ in 1:DAY]
    row = DataFrame(reshape(vcat(x0, k0, t0, comps_ind, starts, free_ind, solve_t), 1, :), :auto)
    CSV.write(file_name, row; append=true, writeheader=false)
end

function get_policy!(cache, file_name::String, x0::Vector{Int}, k0::Int,
                      t0_local::Int, t0_global::Int, solver::Symbol)
    key = (copy(x0), k0, t0_local)
    if haskey(cache, key)
        c, sd, fd, _ = cache[key]
        return c, sd, fd
    end
    comps, sd, fd, solve_t = solver == :cg ?
        solve_RMP_CG(x0, k0, t0_global) :
        solve_relaxed_and_round(x0, k0, t0_global)
    cache[key] = (comps, sd, fd, solve_t)
    append_policy_row(file_name, x0, k0, t0_local, comps, sd, fd, solve_t)
    method_id = solver == :cg ? 1 : 2
    plot_and_save_schedule(comps, sd, fd, method_id, t0_local, "g$(t0_global)")
    return comps, sd, fd
end

# -----------------------------------------------------------------------------
# 8. Main loop: simulate, compute cost, write out, repeat
# -----------------------------------------------------------------------------

function run_method(solver::Symbol, tag::String; n_sim::Int=5)
    policies_file = "policies_25_$(tag).csv"
    cost_file     = "cost_25_$(tag).csv"
    cache = load_cache(policies_file)

    costs = Float64[]
    for sim in 1:n_sim
        x = ones(Int, C)
        k = K0
        total_cost = 0.0
        for tglob in 1:TBAR
            tloc = (tglob - 1) % STAGE_YEAR + 1
            @logmsg("[%s] sim %d/%d — stage %d/%d (season %d)\n", tag, sim, n_sim, tglob, TBAR, tloc)
            comps, sd, fd = get_policy!(cache, policies_file, x, k, tloc, tglob, solver)
            cost, x_new = simulate_stage(x, comps, sd, fd)
            total_cost += cost
            x = x_new
            k = (tglob % STAGE_YEAR == 0) ? K0 : k - length(fd)
        end
        push!(costs, total_cost)
        CSV.write(cost_file, DataFrame(reshape([tag, total_cost], 1, :), :auto);
                  append=true, writeheader=false)
        @printf("[%s] simulation %d/%d — total cost: %.1f\n", tag, sim, n_sim, total_cost)
    end
    @printf("[%s] mean cost over %d simulations: %.2f (σ=%.2f)\n",
            tag, n_sim, mean(costs), std(costs))
    return costs
end

# -----------------------------------------------------------------------------
# 9. Run
# -----------------------------------------------------------------------------

println("=== Method 1: column generation ===")
costs_cg = run_method(:cg, "cg"; n_sim=N_SIM)

println("=== Method 2: direct relaxation + rounding + repair ===")
costs_relax = run_method(:relax, "relax"; n_sim=N_SIM)
