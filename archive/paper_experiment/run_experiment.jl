# run_experiment.jl
#
# Simulate the policy of one or several methods on one instance of the paper (Section 5)
# and store the results in csv files. Usage:
#
#   julia paper_experiment/run_experiment.jl <method>[,<method>...] <instance> [n_scenarios] [--new] [--no-heuristic] [--alpha=<value>]
#
#   method   : fluid_full | fluid_cg | fluid_enum | baseline_greedy | fluid_cg_legacy | fluid_cg_light | fluid_cg_order2 | compact |
#              lift_project | baseline1 | baseline2
#              (fluid_cg, compact, fluid_cg_order2 and lift_project are the LPs of Sections 3.2, 3.3,
#              4.2 and 4.3 of paper.tex, fluid_cg_paper.jl and compact_paper.jl; fluid_cg_legacy,
#              compact_legacy, fluid_cg_order2_legacy and lift_project_legacy are their former
#              implementations)
#              (comma-separated, no spaces, to run several methods back to back, e.g.
#              "fluid_cg,compact,fluid_cg_order2")
#   instance : one_pole | two_poles | three_poles (or their *_simple variants)
#   --new    : delete the previous results of this (method, instance) and start again
#   --no-heuristic : disable heuristic_extraction.jl's shared policy-extraction rule and
#                    fall back to each method's own bespoke extraction (column argmax /
#                    extract_policy_compact); heuristic_extraction is on by default.
#   --gamma=<value> : baseline1 and baseline_greedy: weight γ ∈ [0, 1] of the maintenance days in the per-component
#                    threshold optimisation (baseline1.jl); default 1.0.
#   --alpha=<value> : scales δ (build_fluid_data's `alpha`, fluid_lp_cg.jl), the per-component
#                    failure penalty relative to the (unscaled) replacement cost and joint
#                    schedule cost; default 1.0.
#
# Results (folder paper_experiment/results/), one set of files per method:
#   <method>_<instance>_costs.csv    one row per simulated scenario (total cost, penalties,
#                                     lp_cost, ...)
#   <method>_<instance>_periods.csv  one row per period of each scenario (state, action,
#                                     true cost, lp_cost)
#   <method>_<instance>_solves.csv   one row per new LP solved, with its computing time
#   <method>_<instance>_policies.jld2  cache of the actions already computed
# and:
#   report_<instance>_<timestamp>.txt  mean cost / std dev / CVaR_0.95 / mean lp_cost / total
#                           LP solve time, one row per method. A new file is created for every
#                           run (never overwrites an earlier report), and a method's row is
#                           appended to it as soon as that method's simulation finishes, not
#                           only once every method in this run has finished.
#
# "cost" is the TRUE, non-additive cost of simulate_period (simulation.jl); on the original
# instances it is the true cost Tc of paper.tex (true_period_cost, fluid_lp_cg.jl), with no
# degradation during the period;
# "lp_cost" is the LP's own additive approximation (lp_approx_cost, fluid_lp_cg.jl) of that
# SAME simulated trajectory -- the cost model actually optimised by the method that chose
# the policy (order-1 for fluid_cg/compact/lift_project, order-2 for fluid_cg_order2). The
# two differ whenever a maintained component and a simultaneously-degraded, non-maintained
# component of the same pole overlap on the same day (Appendix C.4 of the paper: "In the
# numerical experiments section we assumed an approximated problem where the cost is the
# sum of subsystem costs and the cost of the joint action. In the reality the cost is
# global."). Keeping both side by side lets this model gap be read directly off the results.

using Printf, Statistics, Dates

include(joinpath(@__DIR__, "instances.jl"))
include(joinpath(@__DIR__, "fluid_lp_cg.jl"))
include(joinpath(@__DIR__, "fluid_cg_paper.jl"))
include(joinpath(@__DIR__, "fluid_enum_paper.jl"))
include(joinpath(@__DIR__, "baseline_greedy_paper.jl"))
include(joinpath(@__DIR__, "heuristic_extraction.jl"))
include(joinpath(@__DIR__, "fluid_cg_light.jl"))
include(joinpath(@__DIR__, "compact_lp.jl"))
include(joinpath(@__DIR__, "compact_paper.jl"))
include(joinpath(@__DIR__, "fluid_cg_order2.jl"))
include(joinpath(@__DIR__, "lift_project.jl"))
include(joinpath(@__DIR__, "baseline1.jl"))
include(joinpath(@__DIR__, "simulation.jl"))

# ── Options (default values, overridden by the command line) ────────────────────
METHODS     = ["fluid_cg"]
INSTANCE    = "one_pole"
N_SCENARIOS = 100
NEW_EXPERIMENT = false
H_FLUID     = 6        # fluid horizon (in periods)
WARM_START_COLUMNS = true
HEURISTIC_EXTRACTION = true   # same method-agnostic policy extraction for every method
                               # (heuristic_extraction.jl), on by default; --no-heuristic
                               # falls back to each method's own bespoke extraction
ALPHA = 1.0                   # scales δ (build_fluid_data's `alpha`), --alpha=<value> to override
# tag of the cost model of the simulator, in the names of the result files: "paper" for the
# former discretised model (QUOTA_STEP = 15, instances.jl), "paperexact" for the exact model of
# paper.tex (QUOTA_STEP = 1), so that the policies cached for one are never reused for the other
MODEL_VERSION = (QUOTA_STEP == 15 ? "paper" : QUOTA_STEP == 1 ? "paperexact" : "paper_q$(QUOTA_STEP)") *
                (MTBF_SCALE == 1.0 ? "" : "_mtbf$(MTBF_SCALE)") *   # MTBF_SCALE: build_MC_utils.jl
                (MOBILIZATION_COST == 0.0 ? "" : "_mob$(MOBILIZATION_COST)") *   # instances.jl
                (get(ENV, "MTBF_FROM_SIMPLE", "0") == "1" ? "_mtbfsimple" : "") *   # instances.jl
                "_start$(FIRST_START_DAY)"   # earliest start day (instances.jl): caches of other models are not reused
GAMMA = 1.0                   # baseline1: weight of the maintenance days in the per-component
                               # policies (baseline1.jl, baseline_greedy_paper.jl), --gamma=<value> to override

let args = filter(a -> !startswith(a, "--"), ARGS)
    length(args) >= 1 && (global METHODS = String.(split(args[1], ',')))
    length(args) >= 2 && (global INSTANCE = args[2])
    length(args) >= 3 && (global N_SCENARIOS = parse(Int, args[3]))
    "--new" in ARGS && (global NEW_EXPERIMENT = true)
    "--no-heuristic" in ARGS && (global HEURISTIC_EXTRACTION = false)
    for a in ARGS
        startswith(a, "--alpha=") && (global ALPHA = parse(Float64, split(a, "=")[2]))
        startswith(a, "--gamma=") && (global GAMMA = parse(Float64, split(a, "=")[2]))
        startswith(a, "--H=") && (global H_FLUID = parse(Int, split(a, "=")[2]))
    end
end

# ── Instance (shared by every method) ────────────────────────────────────────────
instance_dict = Dict(i.name => i for i in instances)
haskey(instance_dict, INSTANCE) || error("unknown instance $INSTANCE; choose among $(collect(keys(instance_dict)))")
fd = build_fluid_data(instance_dict[INSTANCE]; alpha = ALPHA)

not_implemented(name) = (args...; kw...) -> error("method $name is not implemented yet")

# LP-cost model actually optimised by each method: order-1 for every method except
# fluid_cg_order2, which corrects it with the pairwise Δ terms -- must use the SAME
# max_pairs default (3) as make_policy_fluid_cg_order2, so the reported lp_cost matches
# the pairs actually tracked by the policy.
ORDER2_MAX_PAIRS = 3
order1_lp_cost(x, col) = lp_approx_cost(fd, x, col)
function order2_lp_cost_fn()
    Δ_pairs = compute_Δ_pairs(fd; max_pairs = ORDER2_MAX_PAIRS)
    return (x, col) -> lp_approx_cost(fd, x, col; Δ_pairs = Δ_pairs)
end

"Build the (policy, lp_cost) functions of `method`, writing its LP-solve log to `log_solve`."
function make_policy(method::AbstractString, policies_file::String, log_solve::Function)
    method_factory = Dict{String,Function}(
        # Full fluid LP of Section 3.2 of paper.tex (fluid_cg_paper.jl): true cost at the first
        # time step, separable approximation of the paper at t >= 2, argmax policy (Section 3.4).
        # HEURISTIC_EXTRACTION does not apply to it.
        "fluid_cg"        => () -> (make_policy_fluid_cg_paper(fd; H = H_FLUID,
                                                              cache_file = policies_file, log_solve = log_solve),
                                     (x, col) -> paper_approx_cost(fd, x, col)),
        # Same Full fluid LP, solved without column generation, with every joint action
        # enumerated from the start (fluid_enum_paper.jl).
        "fluid_enum"      => () -> (make_policy_fluid_enum_paper(fd; H = H_FLUID,
                                                                cache_file = policies_file, log_solve = log_solve),
                                     (x, col) -> paper_approx_cost(fd, x, col)),
        # Former implementation of the full fluid LP (fluid_lp_cg.jl), kept for comparison.
        "fluid_cg_legacy" => () -> (make_policy_fluid_cg(fd; H = H_FLUID, warm_start = WARM_START_COLUMNS,
                                                        cache_file = policies_file, log_solve = log_solve,
                                                        heuristic_extraction = HEURISTIC_EXTRACTION),
                                     order1_lp_cost),
        "fluid_full"      => not_implemented("fluid_full (full fluid LP without column generation)"),
        "fluid_cg_light"  => () -> (make_policy_fluid_cg_light(fd; H = H_FLUID, warm_start = WARM_START_COLUMNS,
                                                              cache_file = policies_file, log_solve = log_solve,
                                                              heuristic_extraction = HEURISTIC_EXTRACTION),
                                     order1_lp_cost),
        # Order-2 fluid LP of Section 4.2 of paper.tex (fluid_cg_paper.jl), on the pairs of
        # components of a same substation.
        "fluid_cg_order2" => () -> (make_policy_fluid_cg_paper(fd; H = H_FLUID, pairs = same_pole_pairs(fd),
                                                              cache_file = policies_file, log_solve = log_solve),
                                     (x, col) -> paper_approx_cost(fd, x, col; pairs = same_pole_pairs(fd))),
        # Compact fluid LP of Section 3.3 of paper.tex (compact_paper.jl).
        "compact"         => () -> (make_policy_compact_paper(fd; H = H_FLUID,
                                                             cache_file = policies_file, log_solve = log_solve),
                                     (x, col) -> paper_approx_cost(fd, x, col)),
        # Full fluid LP strengthened by the lift-and-project cuts of Section 4.3 of paper.tex
        # (fluid_cg_paper.jl), on the pairs of components of a same substation.
        "lift_project"    => () -> (make_policy_fluid_cg_paper(fd; H = H_FLUID, cuts = true,
                                                              cache_file = policies_file, log_solve = log_solve),
                                     (x, col) -> paper_approx_cost(fd, x, col)),
        # Former implementations, kept for comparison.
        "fluid_cg_order2_legacy" => () -> (make_policy_fluid_cg_order2(fd; H = H_FLUID, warm_start = WARM_START_COLUMNS,
                                                               cache_file = policies_file, log_solve = log_solve,
                                                               max_pairs = ORDER2_MAX_PAIRS,
                                                               heuristic_extraction = HEURISTIC_EXTRACTION),
                                     order2_lp_cost_fn()),
        "compact_legacy"  => () -> (make_policy_compact_lp(fd; H = H_FLUID,
                                                          cache_file = policies_file, log_solve = log_solve,
                                                          heuristic_extraction = HEURISTIC_EXTRACTION),
                                     order1_lp_cost),
        "lift_project_legacy" => () -> (make_policy_lift_project(fd; H = H_FLUID, warm_start = WARM_START_COLUMNS,
                                                            cache_file = policies_file, log_solve = log_solve,
                                                            heuristic_extraction = HEURISTIC_EXTRACTION),
                                     order1_lp_cost),
        "baseline1"       => () -> (make_policy_baseline1(fd; cache_file = policies_file, log_solve = log_solve, gamma = GAMMA),
                                     order1_lp_cost),
        # Thresholds per component without quota (offline), greedy free days (online), on the
        # model of paper.tex (baseline_greedy_paper.jl).
        "baseline_greedy" => () -> (make_policy_baseline_greedy_paper(fd; log_solve = log_solve, gamma = GAMMA),
                                     (x, col) -> paper_approx_cost(fd, x, col)),
        "baseline2"       => not_implemented("baseline2 (PPO)"),
    )
    haskey(method_factory, method) || error("unknown method $method; choose among $(sort(collect(keys(method_factory))))")
    return method_factory[method]()
end

# One new, timestamped report file per run (never overwrites an earlier report); its header is
# written now, and each method's row is appended to it as soon as that method's simulation
# finishes (below), instead of collecting every row in memory and writing them all at the end.
report_file = joinpath(@__DIR__, "results", "report_$(INSTANCE)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).txt")
mkpath(dirname(report_file))
open(report_file, "w") do io
    @printf(io, "Report -- instance=%s  scenarios=%d  fluid horizon=%d  (generated %s)\n\n",
            INSTANCE, N_SCENARIOS, H_FLUID, string(now()))
    @printf(io, "%-18s %12s %12s %12s %12s %18s\n", "method", "mean_cost", "std", "CVaR_0.95", "mean_lp_cost", "total_solve_time_s")
end

for method in METHODS
    # ── Files ─────────────────────────────────────────────────────────────────────
    results_dir   = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    # H_FLUID is part of the prefix (not just the default 6): the cached policies depend on it,
    # and reusing a cache computed for a different fluid horizon would silently be wrong.
    # MODEL_VERSION is also part of the prefix: the simulator now evaluates the true cost of
    # paper.tex (no degradation during the period, true_period_cost), so the results and the
    # policies cached under the former cost model must not be resumed or reused.
    prefix        = joinpath(results_dir, "$(method)_$(INSTANCE)_H$(H_FLUID)_$(MODEL_VERSION)")
    costs_file    = prefix * "_costs.csv"
    periods_file  = prefix * "_periods.csv"
    solves_file   = prefix * "_solves.csv"
    policies_file = prefix * "_policies.jld2"

    if NEW_EXPERIMENT
        for f in (costs_file, periods_file, solves_file, policies_file)
            isfile(f) && (rm(f); @printf("  deleted %s\n", f))
        end
    end

    log_solve(info) = append_csv(solves_file,
        "method,instance,period_in_year,horizon,states,quota,solve_time,lp_value,iterations,n_columns,converged",
        @sprintf("%s,%s,%d,%d,%s,%s,%.4f,%.4f,%d,%d,%d", method, INSTANCE, info.key[3], info.key[4],
                 join(info.key[1], "-"), join(info.key[2], "-"),
                 info.time, info.lp_value, info.iterations, info.n_columns, info.converged))

    policy, lp_cost = make_policy(method, policies_file, log_solve)

    # ── Simulation ────────────────────────────────────────────────────────────────
    done = isfile(costs_file) ? max(0, countlines(costs_file) - 1) : 0   # resume where we stopped

    @printf("\n%s\n", "="^70)
    @printf("SIMULATION  method=%s  instance=%s  scenarios=%d (already done: %d)\n",
            method, INSTANCE, N_SCENARIOS, done)
    @printf("  poles=%d  components=%d  years=%d  fluid horizon=%d\n",
            fd.inst.n_poles, fd.M, fd.inst.n_years, H_FLUID)
    @printf("%s\n", "="^70)

    for scen in (done + 1):N_SCENARIOS
        r = simulate_scenario(fd, policy, scen; method = method, instance = INSTANCE,
                              periods_file = periods_file, lp_cost = lp_cost)
        append_csv(costs_file,
            "method,instance,scenario,total_cost,penalty_cost,replacement_cost,lp_cost,n_maintenances,solve_time,n_solves",
            @sprintf("%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%d,%.4f,%d", method, INSTANCE, scen,
                     r.total, r.penalty, r.replacement, r.lp_cost, r.n_maintenances, r.solve_time, r.n_solves))
        @printf("  scenario %3d/%d  cost=%.2f (lp_cost=%.2f)  (penalty %.2f + replacement %.2f)  new LPs=%d  solve time=%.1fs\n",
                scen, N_SCENARIOS, r.total, r.lp_cost, r.penalty, r.replacement, r.n_solves, r.solve_time)
    end

    # ── Summary ───────────────────────────────────────────────────────────────────
    lines = readlines(costs_file)[2:end]
    costs    = [parse(Float64, split(l, ",")[4]) for l in lines]
    lp_costs = [parse(Float64, split(l, ",")[7]) for l in lines]
    solve_times = isfile(solves_file) ? [parse(Float64, split(l, ",")[7]) for l in readlines(solves_file)[2:end]] : Float64[]
    mean_cost    = mean(costs)
    std_cost     = length(costs) > 1 ? std(costs) : 0.0
    cvar_cost    = cvar(costs)
    mean_lp_cost = mean(lp_costs)
    total_time   = sum(solve_times; init = 0.0)
    @printf("\n%s\n", "="^70)
    @printf("  method             : %s\n", method)
    @printf("  Mean cost          : %.2f\n", mean_cost)
    @printf("  Std dev            : %.2f\n", std_cost)
    @printf("  CVaR 0.95          : %.2f\n", cvar_cost)
    @printf("  Mean lp_cost       : %.2f  (model gap: %.2f)\n", mean_lp_cost, mean_cost - mean_lp_cost)
    @printf("  LPs solved         : %d  (total %.1fs, mean %.2fs)\n", length(solve_times),
            total_time, isempty(solve_times) ? 0.0 : mean(solve_times))
    @printf("%s\n", "="^70)

    open(report_file, "a") do io
        @printf(io, "%-18s %12.2f %12.2f %12.2f %12.2f %18.1f\n",
                method, mean_cost, std_cost, cvar_cost, mean_lp_cost, total_time)
    end
    @printf("  (report row appended to %s)\n", report_file)
end

@printf("\nReport written to %s\n", report_file)
