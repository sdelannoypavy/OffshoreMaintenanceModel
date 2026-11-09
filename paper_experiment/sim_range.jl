# simulate scenarios first..last of one policy on one instance. Writes <prefix>_periods.csv (one row per period) and
# <prefix>_solves.csv (one row per new decision: its total time, and the time of the pricing MILPs and of the LP
# solves, the rest being the construction of the models)
const D = "C:/Users/sodel/MultiHorizonMaintenanceModel/paper_experiment"
include(joinpath(D, "instances.jl")); include(joinpath(D, "fluid_lp_cg.jl"))
include(joinpath(D, "fluid_cg_paper.jl")); include(joinpath(D, "fluid_enum_paper.jl"))
include(joinpath(D, "compact_paper.jl")); include(joinpath(D, "baseline_greedy_paper.jl")); include(joinpath(D, "simulation.jl"))
using Printf
# limit the threads of Gurobi (several simulations in parallel)
haskey(ENV, "GUROBI_THREADS") && Gurobi.GRBsetintparam(GRB_ENV, "Threads", parse(Int, ENV["GUROBI_THREADS"]))
method = ARGS[1]; name = ARGS[2]; first = parse(Int, ARGS[3]); last = parse(Int, ARGS[4]); prefix = ARGS[5]
fd = build_fluid_data(Dict(i.name => i for i in instances)[name])
periods = prefix * "_periods.csv"
# resume: keep the scenarios already complete (120 periods), drop the rows of an interrupted scenario
done_scen = Set{Int}()
if isfile(periods)
    rows = [l for l in readlines(periods) if length(split(l, ",")) >= 13]   # drop a line cut by an interruption
    cnt = Dict{Int,Int}()
    for l in rows[2:end]; s = parse(Int, split(l, ",")[3]); cnt[s] = get(cnt, s, 0) + 1; end
    done_scen = Set(s for (s, c) in cnt if c == 120)
    kept = [rows[1]; [l for l in rows[2:end] if parse(Int, split(l, ",")[3]) in done_scen]]
    length(kept) > 1 ? write(periods, join(kept, "\n") * "\n") : rm(periods)
    println("resume: $(length(done_scen)) scenarios already complete"); flush(stdout)
end
solves = prefix * "_solves.csv"
log_solve(info) = append_csv(solves, "t0_cyc,horizon,time,t_milp,t_lp",
                             @sprintf("%d,%d,%.4f,%.4f,%.4f", info.key[3], info.key[4], info.time, info.t_milp, info.t_lp))
# Policy cache on disk: every decision (state, period of the year, horizon) -> action is written to
# <prefix>_policy.csv (one row per decision, see load_cache_csv in fluid_lp_cg.jl) as soon as it is computed, and
# reloaded at restart, so that no decision is computed twice, even in an interrupted scenario. The decisions of the
# other chunks of the same method, instance and model (<dir>/<instance>_<method>_<chunk>_policy.csv, same QUOTA_STEP
# and FIRST_START_DAY since the prefix encodes the folder of the final model) are copied into this chunk's file at
# start and again before each scenario, each process writing only to its own file.
cache_file = prefix * "_policy.csv"
known = load_cache(cache_file)[1]   # all the decisions known to this process: its own and those of the other chunks
const NO_ID = Ref(0)
function merge_other_chunks!()
    dir = dirname(cache_file); pat = Regex("^" * name * "_" * method * raw"_\d+_policy\.csv$")
    added = 0
    for f in readdir(dir)
        occursin(pat, f) || continue
        p = joinpath(dir, f); p == cache_file && continue
        # read a copy, so that the other process is not prevented from writing to its file (Windows)
        tmp = tempname() * ".csv"
        other = try cp(p, tmp; force = true); redirect_stdout(() -> load_cache(tmp)[1], devnull) catch; Dict() end
        rm(tmp; force = true)
        for (k, v) in other
            haskey(known, k) && continue
            known[k] = v; append_cache_entry!(cache_file, NO_ID, k, v); added += 1
        end
    end
    return added
end
let added = merge_other_chunks!()
    println("policy cache: $(length(known)) decisions ($added copied from the other chunks)"); flush(stdout)
end
pol0 = method == "fluid_enum"   ? make_policy_fluid_enum_paper(fd; H = 6, cache_file = cache_file, log_solve = log_solve) :
      method == "quota_sub"    ? make_policy_fluid_enum_paper(fd; H = 6, quota_subsystem = true, cache_file = cache_file, log_solve = log_solve) :
      method == "fluid_cg"     ? make_policy_fluid_cg_paper(fd; H = 6, cache_file = cache_file, log_solve = log_solve) :
      method == "compact"      ? make_policy_compact_paper(fd; H = 6, cache_file = cache_file, log_solve = log_solve) :
      method == "compact_kd"   ? make_policy_compact_paper(fd; H = 6, kappa_dist = true, cache_file = cache_file, log_solve = log_solve) :
      method == "lift_project" ? make_policy_fluid_cg_paper(fd; H = 6, cuts = true, cache_file = cache_file, log_solve = log_solve) :
      method == "order2"       ? make_policy_fluid_cg_paper(fd; H = 6, pairs = same_pole_pairs(fd), cache_file = cache_file, log_solve = log_solve) :
      method == "baseline_greedy" ? make_policy_baseline_greedy_paper(fd; gamma = parse(Float64, get(ENV, "BASELINE_GAMMA", "1.0")), log_solve = log_solve) : error("unknown method $method")
# a decision already known (computed by this process or copied from another chunk) is not computed again
T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
function pol(x::Vector{Int}, k::Vector{Int}, t_global::Int)
    method == "baseline_greedy" && return pol0(x, k, t_global)
    key = (Tuple(x), Tuple(k), (t_global - 1) % N_PERIODS_PER_YEAR + 1, min(6, T_total - t_global + 1))
    haskey(known, key) && return Column(known[key][1], known[key][2]), 0.0
    # <prefix>_solving.txt exists while a new decision is computed, so that an external watchdog can stop the runs
    # when a single LP lasts too long (its modification time is the start of the solve)
    write(prefix * "_solving.txt", string(key))
    col, t = pol0(x, k, t_global)
    rm(prefix * "_solving.txt"; force = true)
    known[key] = (col.start, col.free_days, t, NaN)
    return col, t
end
@printf("%s %s scenarios %d..%d  MOB=%.2f  MTBF_SCALE=%.2f  QUOTA_STEP=%d\n", method, name, first, last, MOBILIZATION_COST, MTBF_SCALE, QUOTA_STEP); flush(stdout)
for s in first:last
    s in done_scen && continue
    if method != "baseline_greedy"
        added = merge_other_chunks!()
        added > 0 && (println("policy cache: $added decisions copied from the other chunks"); flush(stdout))
    end
    t = time()
    r = simulate_scenario(fd, pol, s; method = method, instance = name, periods_file = periods)
    @printf("scenario %d (%d..%d)  cost %.2f  maintenances %d  %.0fs\n", s, first, last, r.total, r.n_maintenances, time() - t); flush(stdout)
end
println("done")
