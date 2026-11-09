# conflicts9.jl
#
# Cheap diagnostic of the interactions between substations of interactions9.jl. For S = 2..5 and each scenario,
# every substation is simulated ALONE under the policy of the Full fluid LP + column generation of one substation,
# with the draws of the instance with S substations (same draws as interactions9.jl). We count the periods in
# which the union of the actions of the S substations exceeds the workforce R on some day. If there is no such
# period, the decomposed policy is feasible for the S substations, and the shared workforce never binds.
#
#   QUOTA_STEP=15 GUROBI_THREADS=2 julia --project=. paper_experiment/conflicts9.jl <n_scenarios> <outdir>

nscen = parse(Int, ARGS[1]); outdir = ARGS[2]
Srange = length(ARGS) >= 4 ? (parse(Int, ARGS[3]):parse(Int, ARGS[4])) : (2:5)
# component data of interactions9.jl (its lines before the construction of the instances)
src = readlines(joinpath(@__DIR__, "interactions9.jl"))
stop = findfirst(l -> startswith(l, "fdS ="), src)
src = filter(l -> !startswith(l, "S = parse"), src[1:stop-1])
include_string(Main, join(src, "\n"))

fd1 = build_fluid_data(make_inst9(1))
cache = joinpath(outdir, "conflicts_S1_policy.csv")
!isfile(cache) && isfile(joinpath(outdir, "S1_1_policy.csv")) && cp(joinpath(outdir, "S1_1_policy.csv"), cache)
pol1 = make_policy_fluid_cg_paper(fd1; H = 6, cache_file = cache)
T_total = 20 * N_PERIODS_PER_YEAR

"Actions of one substation alone over the scenario, with the draws U[t, τ, m] of its 9 components."
function actions(U)
    x = ones(Int, 9); k = copy(fd1.K0); cols = Column[]; total = 0.0
    for t in 1:T_total
        (t - 1) % N_PERIODS_PER_YEAR == 0 && (k = copy(fd1.K0))
        col, _ = pol1(x, k, t); push!(cols, col)
        pen, rep, x = simulate_period(fd1, x, col, U[t, :, :]); total += pen + rep
        k = k .- length.(col.free_days)
    end
    return cols, total
end
demand(col, τ) = sum(fd1.rho[m] for m in 1:9 if col.start[m] > 0 && col.start[m] <= τ <= col.start[m] + fd1.d[m] - 1;
                     init = 0.0)

out = joinpath(outdir, "conflicts.csv")
# resume: the (S, scenario) pairs already in the output file are not computed again
done = Set{Tuple{Int,Int}}()
if isfile(out)
    for l in readlines(out)[2:end]; r = split(l, ","); push!(done, (parse(Int, r[1]), parse(Int, r[2]))); end
else
    open(io -> println(io, "S,scenario,cost_sum_alone,maint,periods_two_or_more_subs_maintained,periods_conflict"), out, "w")
end
for S in Srange, s in 1:nscen
    (S, s) in done && continue
    U = rand(MersenneTwister(1000 + s), T_total, T_PERIOD, 9S)
    res = [actions(U[:, :, (9(p-1)+1):(9p)]) for p in 1:S]
    nm = sum(count(c -> any(>(0), c.start), r[1]) for r in res)
    multi = 0; conf = 0
    for t in 1:T_total
        act = [r[1][t] for r in res]
        count(c -> any(>(0), c.start), act) >= 2 && (multi += 1)
        any(sum(demand(c, τ) for c in act) > 1.0 + 1e-9 for τ in 1:T_PERIOD) && (conf += 1)
    end
    open(io -> println(io, "$S,$s,$(round(sum(r[2] for r in res), digits = 4)),$nm,$multi,$conf"), out, "a")
end
for S in Srange
    rows = [split(l, ",") for l in readlines(out)[2:end] if startswith(l, "$S,")]
    @printf("S=%d  scenarios %d  periods with >=2 substations maintained %d  periods with a workforce conflict %d  scenarios with a conflict %d\n",
            S, length(rows), sum(parse(Int, r[5]) for r in rows), sum(parse(Int, r[6]) for r in rows),
            count(r -> parse(Int, r[6]) > 0, rows))
end
