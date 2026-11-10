# resched9.jl
#
# Can the workforce conflicts found by conflicts9.jl be resolved at no cost? For S = 2..5 and each scenario, every
# substation is simulated alone under the policy of the Full fluid LP + CG of one substation (as in conflicts9.jl).
# In each period where the union of the actions exceeds the workforce R, we search, by enumeration of the start days
# (grid FIRST_START_DAY, FIRST_START_DAY + QUOTA_STEP, ...), a schedule of the SAME maintained components such that
#   - the daily workforce demand is at most R,
#   - every substation uses at most as many penalty-free days as alone (so that its future quota is unchanged),
#   - the total penalty of the non-free maintenance days is at most the total penalty of the actions alone.
# If such a schedule exists, the conflict is absorbed at no cost: the decomposed policy can be repaired into a policy
# of the S substations with the same cost, so that cost(S) <= sum of the costs alone for this policy.
#
#   QUOTA_STEP=15 GUROBI_THREADS=2 julia --project=. paper_experiment/resched9.jl <n_scenarios> <outdir>

nscen = parse(Int, ARGS[1]); outdir = ARGS[2]
Srange = length(ARGS) >= 4 ? (parse(Int, ARGS[3]):parse(Int, ARGS[4])) : (2:5)
src = readlines(joinpath(@__DIR__, "interactions9.jl"))
stop = findfirst(l -> startswith(l, "fdS ="), src)
src = filter(l -> !startswith(l, "S = parse"), src[1:stop-1])
include_string(Main, join(src, "\n"))

fd1 = build_fluid_data(make_inst9(1))
cache = joinpath(outdir, "resched_S1_policy.csv")   # copy of the cache of conflicts9.jl (which may still be running)
cp(joinpath(outdir, "conflicts_S1_policy.csv"), cache; force = true)
pol1 = make_policy_fluid_cg_paper(fd1; H = 6, cache_file = cache)
T_total = 20 * N_PERIODS_PER_YEAR

function actions(U)
    x = ones(Int, 9); k = copy(fd1.K0); cols = Column[]; ks = Int[]
    for t in 1:T_total
        (t - 1) % N_PERIODS_PER_YEAR == 0 && (k = copy(fd1.K0))
        col, _ = pol1(x, k, t); push!(cols, col); push!(ks, k[1])
        _, _, x = simulate_period(fd1, x, col, U[t, :, :])
        k = k .- length.(col.free_days)
    end
    return cols, ks
end
busy(starts, ds) = [any(s > 0 && s <= τ <= s + d - 1 for (s, d) in zip(starts, ds)) for τ in 1:T_PERIOD]
"Penalty-free days used and penalty of one substation: free days cover the maintenance days, up to the quota k (blocks of QUOTA_STEP)."
function free_pen(nbusy, k)
    free = min(nbusy, k); return free, nbusy - free
end

grid = collect(FIRST_START_DAY:QUOTA_STEP:T_PERIOD)
out = joinpath(outdir, "resched_S$(first(Srange))-$(last(Srange)).csv")
open(io -> println(io, "S,scenario,period,n_maintained,resolved"), out, "w")
for S in Srange, s in 1:nscen
    U = rand(MersenneTwister(1000 + s), T_total, T_PERIOD, 9S)
    res = [actions(U[:, :, (9(p-1)+1):(9p)]) for p in 1:S]
    for t in 1:T_total
        act = [r[1][t] for r in res]
        dem(τ) = sum(fd1.rho[m] for c in act for m in 1:9 if c.start[m] > 0 && c.start[m] <= τ <= c.start[m] + fd1.d[m] - 1; init = 0.0)
        any(dem(τ) > 1 + 1e-9 for τ in 1:T_PERIOD) || continue
        # maintained components (substation, component) and their data
        items = [(p, m) for p in 1:S for m in 1:9 if act[p].start[m] > 0]
        free_alone = [length(act[p].free_days[1]) for p in 1:S]
        # penalty of the actions alone: the maintenance days that are not declared free (a substation may declare more
        # free days than it has maintenance days, so this is not the number of maintenance days minus the free days)
        pen_alone = sum(column_penalty(fd1, act[p]) for p in 1:S)
        opts = [[g for g in grid if g + fd1.d[m] - 1 <= T_PERIOD] for (p, m) in items]
        found = false
        for choice in Iterators.product(opts...)
            ok = all(sum(fd1.rho[m] for ((p, m), g) in zip(items, choice) if g <= τ <= g + fd1.d[m] - 1; init = 0.0) <= 1 + 1e-9
                     for τ in 1:T_PERIOD)
            ok || continue
            pen = 0; fine = true
            for p in 1:S
                st = zeros(Int, 9); for ((q, m), g) in zip(items, choice); q == p && (st[m] = g); end
                f, pe = free_pen(count(busy(st, fd1.d)), min(res[p][2][t], free_alone[p]))
                pen += pe
            end
            if pen <= pen_alone; found = true; break; end
        end
        open(io -> println(io, "$S,$s,$t,$(length(items)),$(found ? 1 : 0)"), out, "a")
    end
end
for S in Srange
    rows = [split(l, ",") for l in readlines(out)[2:end] if startswith(l, "$S,")]
    nres = count(r -> r[5] == "1", rows)
    @printf("S=%d  conflict periods %d  resolved at no cost %d  unresolved %d (scenarios %s)\n", S, length(rows), nres,
            length(rows) - nres, join(unique([r[2] for r in rows if r[5] == "0"]), " "))
end
