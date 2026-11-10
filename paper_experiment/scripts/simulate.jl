# simulate.jl
#
# Simulates the policy of one method of paper.tex on the scenarios first..last of one instance.
# Usage (from the root of the repository):
#
#   GUROBI_THREADS=2 julia --project=. paper_experiment/scripts/simulate.jl <method> <instance> <first> <last> [outdir] [--gamma=<γ>]
#
#   method   : 2SB | CFLP | JFLP-enum | JFLP-CG | JFLP-cuts | JFLP-2 | JFLP-Q
#   instance : one_pole | two_poles | three_poles (full model), one_pole_simple | two_poles_simple |
#              three_poles_simple (simplified model)
#   outdir   : folder of the result files, paper_experiment/results by default
#   --gamma  : parameter γ of 2SB (scripts/tune_gamma.jl), 1 by default
#
# Writes, with prefix <outdir>/<instance>_<method>_<first>:
#   _periods.csv : one row per simulated period (state, action, cost, computing time of the decision),
#   _solves.csv  : one row per new decision, with its total computing time and the time of the pricing
#                  MILPs (t_milp) and of the LP solves (t_lp), the rest being the construction of the models,
#   _policy.csv  : the decisions computed (cache, see make_fluid_policy in src/model.jl).
# The simulation can be stopped and resumed: the complete scenarios are kept, and the decisions already
# computed are read back. Several ranges of scenarios of the same method and instance can be simulated
# in parallel in the same folder: each process also reads the decisions of the others.

include(joinpath(@__DIR__, "..", "src", "setup.jl"))

args = filter(a -> !startswith(a, "--"), ARGS)
method, instance = args[1], args[2]
first_scen, last_scen = parse(Int, args[3]), parse(Int, args[4])
outdir = length(args) >= 5 ? args[5] : joinpath(@__DIR__, "..", "results")
i_gamma = findfirst(a -> startswith(a, "--gamma="), ARGS)
gamma = isnothing(i_gamma) ? 1.0 : parse(Float64, split(ARGS[i_gamma], "=")[2])
method in METHODS || error("unknown method $method, expected one of $METHODS")

fd = build_fluid_data(INSTANCES[instance])
prefix = joinpath(outdir, "$(instance)_$(method)_$(first_scen)")
periods_file = prefix * "_periods.csv"; solves_file = prefix * "_solves.csv"; cache_file = prefix * "_policy.csv"
T_total = fd.inst.n_years * N_PERIODS_PER_YEAR

# resume: keep the complete scenarios (T_total periods), drop the rows of an interrupted scenario
done = Set{Int}()
if isfile(periods_file)
    rows = [l for l in readlines(periods_file) if length(split(l, ",")) == length(split(PERIODS_HEADER, ","))]
    count_of = Dict{Int,Int}()
    for l in rows[2:end]; s = parse(Int, split(l, ",")[3]); count_of[s] = get(count_of, s, 0) + 1; end
    done = Set(s for (s, c) in count_of if c == T_total)
    kept = [rows[1]; [l for l in rows[2:end] if parse(Int, split(l, ",")[3]) in done]]
    length(kept) > 1 ? write(periods_file, join(kept, "\n") * "\n") : rm(periods_file)
    println("resume: $(length(done)) scenarios already complete")
end

log_solve(info) = append_csv(solves_file, "t0_cyc,horizon,time,t_milp,t_lp",
                             @sprintf("%d,%d,%.4f,%.4f,%.4f", info.key[3], info.key[4], info.time, info.t_milp, info.t_lp))
policy = make_policy(method, fd; gamma = gamma, cache_file = cache_file, log_solve = log_solve)

# decisions computed by the other processes of the same method and instance in `outdir`
known = load_cache(cache_file)
function read_other_caches!()
    pattern = Regex("^" * instance * "_" * method * raw"_\d+_policy\.csv$")
    for f in readdir(outdir)
        (occursin(pattern, f) && joinpath(outdir, f) != cache_file) || continue
        tmp = tempname() * ".csv"   # read a copy, so that the other process can keep writing to its file (Windows)
        other = try cp(joinpath(outdir, f), tmp; force = true); redirect_stdout(() -> load_cache(tmp), devnull) catch; Dict() end
        rm(tmp; force = true)
        for (key, value) in other
            haskey(known, key) || (known[key] = value; append_cache_entry!(cache_file, key, value))
        end
    end
end
function shared_policy(x::Vector{Int}, k::Vector{Int}, t::Int)
    method == "2SB" && return policy(x, k, t)   # no LP: nothing to share
    key = decision_key(x, k, (t - 1) % N_PERIODS_PER_YEAR + 1, min(6, T_total - t + 1))
    haskey(known, key) && return Column(known[key][1], known[key][2]), 0.0
    col, dt = policy(x, k, t)
    known[key] = (col.start, col.free_days, dt, NaN)
    return col, dt
end

@printf("%s %s scenarios %d..%d\n", method, instance, first_scen, last_scen); flush(stdout)
for s in first_scen:last_scen
    s in done && continue
    method == "2SB" || read_other_caches!()
    t = time()
    r = simulate_scenario(fd, shared_policy, s; method = method, instance = instance, periods_file = periods_file)
    @printf("scenario %d  cost %.2f  maintenances %d  %.0fs\n", s, r.total, r.n_maintenances, time() - t); flush(stdout)
end
println("done")
