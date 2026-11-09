# interactions9.jl
#
# Search for interactions between substations on the FULL model (free-day quota), with 9 components per
# substation and the MTBF of Table III of PSCC_article.pdf, under the policy of the Full fluid LP + column
# generation (fluid_cg_paper.jl). Usage:
#
#   QUOTA_STEP=15 GUROBI_THREADS=2 julia --project=. paper_experiment/interactions9.jl <S> <first> <last> <outdir>
#
# Components of a substation (MTBF in years): converter 588, water circuit 43, fans 43, pumps 43,
# transformer 3333, converter cooling 122, busbar 4762, second pump system 43 (same data as the pumps),
# fire protection 20 (3 states, full capacity in states 1 and 2, capacity 0 in state 3).
#
# For each scenario, the instance with S substations is simulated, and each of its substations is also
# simulated ALONE (instance with one substation) with exactly the same uniform draws (common random numbers).
# Without interaction, cost(S substations) = sum of the costs of the substations alone on every scenario. The
# only coupling between substations is the shared workforce R.
# Output <outdir>/S<S>_<first>.csv, one row per scenario.

const D = "C:/Users/sodel/MultiHorizonMaintenanceModel/paper_experiment"
include(joinpath(D, "instances.jl")); include(joinpath(D, "fluid_lp_cg.jl"))
include(joinpath(D, "fluid_cg_paper.jl")); include(joinpath(D, "simulation.jl"))
using Printf, Random, Statistics
haskey(ENV, "GUROBI_THREADS") && Gurobi.GRBsetintparam(GRB_ENV, "Threads", parse(Int, ENV["GUROBI_THREADS"]))

S = parse(Int, ARGS[1]); first = parse(Int, ARGS[2]); last = parse(Int, ARGS[3]); outdir = ARGS[4]
mkpath(outdir)

# ── 9-component substation ──────────────────────────────────────────────────
# environment variable MTBF_ALL (years): every component gets this MTBF (e.g. MTBF_ALL=20)
# environment variable DISTINCT=1: the components are made pairwise different, the MTBF being spread linearly
# from 0.8 to 1.2 times its value and the replacement costs being multiplied by 1, 1.01, ..., 1.08 (to break the
# symmetries between identical components, e.g. the two pump systems, or every component with MTBF_ALL)
const DISTINCT = get(ENV, "DISTINCT", "0") == "1"
const MTBF9 = (haskey(ENV, "MTBF_ALL") ? fill(parse(Float64, ENV["MTBF_ALL"]), 9) :
               [588.0, 43.0, 43.0, 43.0, 3333.0, 122.0, 4762.0, 43.0, 20.0]) .*
              (DISTINCT ? collect(range(0.8, 1.2; length = 9)) : ones(9))
println("MTBF of the 9 components: ", MTBF9)
const N9 = [12, 3, 8, 3, 3, 3, 3, 3, 3]
const SHARE9 = [1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5]
const DUR9 = [30, 15, 15, 15, 15, 15, 15, 15, 15]
const COST9 = [5.0, 0.25, 0.125, 0.25, 10.0, 0.5, 0.25, 0.25, 0.25] .* (DISTINCT ? (1 .+ 0.01 .* (0:8)) : ones(9))
# environment variable PARAMSET=heavy_pumps: MTBF of PSCC_article.pdf except the second pump system (20 years) and
# the fire protection (20 years), replacement costs 2 for the converter, 3 for the transformer and 0 for the fire
# protection, and the two pump systems need the whole workforce (100%) for 30 days
if get(ENV, "PARAMSET", "") in ("heavy_pumps", "heavy_pumps90")
    MTBF9 .= [588.0, 43.0, 43.0, 43.0, 3333.0, 122.0, 4762.0, 20.0, 20.0]
    COST9[1] = 2.0; COST9[5] = 3.0; COST9[9] = 0.0
    SHARE9[4] = 1.0; SHARE9[8] = 1.0
    DUR9[4] = 30; DUR9[8] = 30
    println("PARAMSET heavy_pumps: MTBF ", MTBF9, "  costs ", COST9, "  shares ", SHARE9, "  durations ", DUR9)
end
println("replacement costs of the 9 components: ", COST9)

function pump_matrix(MTBF)
    p = 9.55e-5 * 43 / MTBF            # 9.55e-5 gives MTBF 43 (build_MC_utils.jl)
    [1-2p*(1-p) 0 0; 2p*(1-p) 1-p 0; 0 p 1]
end
function trans_matrix(MTBF)
    p = 1 / (365 * MTBF)
    [1-2p*(1-p) 0 0; 2p*(1-p) 1-p 0; 0 p 1]
end
Pd9 = [build_P_converter(MTBF9[1], N9[1]), build_P_linear_daily(N9[2], MTBF9[2]), build_P_fan(MTBF9[3]),
       pump_matrix(MTBF9[4]), trans_matrix(MTBF9[5]), build_P_linear_daily(N9[6], MTBF9[6]),
       build_P_linear_daily(N9[7], MTBF9[7]), pump_matrix(MTBF9[8]), build_P_linear_daily(N9[9], MTBF9[9])]
nmax = maximum(N9)
Pevac9 = zeros(9, nmax)
Pevac9[1:7, :] .= P_evac[1:7, 1:nmax]
Pevac9[8, :] .= P_evac[4, 1:nmax]                     # second pump system: same as the pumps
Pevac9[9, 1:2] .= 100.0                               # fire protection: capacity 100, 100, 0
# PARAMSET=heavy_pumps90: heavy_pumps, and the two pump systems have a capacity of 90% in their degraded state 2
get(ENV, "PARAMSET", "") == "heavy_pumps90" && (Pevac9[4, 2] = 90.0; Pevac9[8, 2] = 90.0)
make_pole9() = Pole(copy(N9), copy(DUR9), deepcopy(Pd9), [P^T_PERIOD for P in Pd9], copy(Pevac9),
                    copy(SHARE9), copy(COST9), FREE_QUOTA, copy(DAILY_PENALTY))
make_inst9(np) = Instance("nine_$(np)", np, [make_pole9() for _ in 1:np], 1.0, T_PERIOD, MARGIN, 20, false)

fdS = build_fluid_data(make_inst9(S)); fd1 = build_fluid_data(make_inst9(1))
solves = joinpath(outdir, "S$(S)_$(first)_solves.csv")
logS(info) = append_csv(solves, "S,t0_cyc,horizon,time,t_milp,t_lp",
                        @sprintf("%d,%d,%d,%.4f,%.4f,%.4f", S, info.key[3], info.key[4], info.time, info.t_milp, info.t_lp))
log1(info) = append_csv(solves, "S,t0_cyc,horizon,time,t_milp,t_lp",
                        @sprintf("%d,%d,%d,%.4f,%.4f,%.4f", 1, info.key[3], info.key[4], info.time, info.t_milp, info.t_lp))
polS = make_policy_fluid_cg_paper(fdS; H = 6, cache_file = joinpath(outdir, "S$(S)_$(first)_policy.csv"), log_solve = logS)
pol1 = S == 1 ? polS :
       make_policy_fluid_cg_paper(fd1; H = 6, cache_file = joinpath(outdir, "S1for$(S)_$(first)_policy.csv"), log_solve = log1)

"Simulation of simulate_scenario (simulation.jl) with given draws U[t, τ, m]."
function run(fd, policy, U)
    x = ones(Int, fd.M); k = copy(fd.K0); total = 0.0; nmaint = 0; tsolve = 0.0
    for t in 1:(fd.inst.n_years * N_PERIODS_PER_YEAR)
        (t - 1) % N_PERIODS_PER_YEAR == 0 && (k = copy(fd.K0))
        col, dt = policy(x, k, t); tsolve += dt
        check_action(fd, col, k)
        nmaint += count(>(0), col.start)
        pen, rep, x = simulate_period(fd, x, col, U[t, :, :])
        total += pen + rep
        k = k .- length.(col.free_days)
    end
    return total, nmaint, tsolve
end

out = joinpath(outdir, "S$(S)_$(first).csv")
done = Set{Int}()
isfile(out) && for l in readlines(out)[2:end]; push!(done, parse(Int, split(l, ",")[1])); end
T_total = 20 * N_PERIODS_PER_YEAR
@printf("S=%d scenarios %d..%d  QUOTA_STEP=%d  (%d already done)\n", S, first, last, QUOTA_STEP, length(done)); flush(stdout)
for s in first:last
    s in done && continue
    t0 = time()
    U = rand(MersenneTwister(1000 + s), T_total, T_PERIOD, fdS.M)
    cS, mS, tS = run(fdS, polS, U)
    alone = [run(fd1, pol1, U[:, :, (9(p-1)+1):(9p)]) for p in 1:S]
    csum = sum(a[1] for a in alone); msum = sum(a[2] for a in alone)
    append_csv(out, "scenario,cost_S,sum_alone,diff,maint_S,maint_alone,solve_S,solve_alone," *
               join(["alone_$p" for p in 1:S], ","),
               @sprintf("%d,%.4f,%.4f,%.4f,%d,%d,%.2f,%.2f,", s, cS, csum, cS - csum, mS, msum, tS,
                        sum(a[3] for a in alone)) * join([@sprintf("%.4f", a[1]) for a in alone], ","))
    @printf("scenario %d  cost S %.3f  sum alone %.3f  diff %+.4f  maint %d/%d  %.0fs\n", s, cS, csum, cS - csum, mS, msum,
            time() - t0); flush(stdout)
end
println("done")
