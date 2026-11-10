# anticipative.jl
#
# Cost of the best anticipative policy on the scenarios first..last of one instance (src/anticipative.jl).
# Usage (from the root of the repository):
#
#   GUROBI_THREADS=2 julia --project=. paper_experiment/scripts/anticipative.jl <instance> <first> <last> [outdir]
#
# Writes <outdir>/<instance>_anticipative_<first>.csv (outdir = paper_experiment/results by default),
# one row per scenario: scenario, anticipative_cost, solve_time, mip_gap. The scenarios already in the
# file are skipped, so that the computation can be stopped and resumed.

include(joinpath(@__DIR__, "..", "src", "setup.jl"))

instance = ARGS[1]; first_scen = parse(Int, ARGS[2]); last_scen = parse(Int, ARGS[3])
outdir = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "..", "results")
out = joinpath(outdir, "$(instance)_anticipative_$(first_scen).csv")

fd = build_fluid_data(INSTANCES[instance])
done = Set{Int}()
isfile(out) && for l in readlines(out)[2:end]; push!(done, parse(Int, split(l, ",")[1])); end
solve = make_anticipative_solver(fd)
for s in first_scen:last_scen
    s in done && continue
    cost, t, gap = solve(s)
    append_csv(out, "scenario,anticipative_cost,solve_time,mip_gap", @sprintf("%d,%.4f,%.3f,%.2e", s, cost, t, gap))
    @printf("scenario %4d  anticipative cost %.4f  (%.1fs)\n", s, cost, t); flush(stdout)
end
