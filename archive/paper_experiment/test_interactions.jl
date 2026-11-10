# test_interactions.jl
#
# Are there interactions between two substations (poles) under baseline 1, on the ORIGINAL
# model (free-day quota, within-period degradation)? Usage:
#
#   julia --project=. paper_experiment/test_interactions.jl <mtbf> <cc_scale> <n_scenarios>
#
#   mtbf     : paper (MTBFs of build_MC_utils.jl) | simple (MTBF_SIMPLE of instances.jl)
#   cc_scale : multiplies every replacement cost (1.0 = paper's costs)
#
# For each scenario, the two-pole instance is simulated, and each of its two poles is also
# simulated ALONE (one-pole instance) with exactly the same uniform draws as in the two-pole
# run (common random numbers). Without interaction, cost(two poles) = cost(pole 1 alone) +
# cost(pole 2 alone) on every scenario; the only coupling is the shared workforce R.

using Printf, Statistics, Random

include(joinpath(@__DIR__, "instances.jl"))
include(joinpath(@__DIR__, "fluid_lp_cg.jl"))
include(joinpath(@__DIR__, "heuristic_extraction.jl"))
include(joinpath(@__DIR__, "fluid_cg_light.jl"))
include(joinpath(@__DIR__, "compact_lp.jl"))
include(joinpath(@__DIR__, "baseline1.jl"))
include(joinpath(@__DIR__, "simulation.jl"))

MTBF_MODE = length(ARGS) >= 1 ? ARGS[1] : "paper"
CC_SCALE  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
N_SCEN    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
METHOD    = length(ARGS) >= 4 ? ARGS[4] : "compact"     # compact | baseline1

function make_pole()
    Pd = MTBF_MODE == "simple" ? daily_matrices_simple() : deepcopy(P_daily)
    Pp = [Pc^T_PERIOD for Pc in Pd]
    return Pole(copy(n), copy(REPAIR_DURATION), Pd, Pp, copy(P_evac), copy(TECHNICIAN_SHARE),
                CC_SCALE .* REPLACEMENT_COST, FREE_QUOTA, copy(DAILY_PENALTY))
end
make_inst(np) = Instance("test_$(np)", np, [make_pole() for _ in 1:np], 1.0, T_PERIOD, MARGIN, 20, false)

fd1 = build_fluid_data(make_inst(1))
fd2 = build_fluid_data(make_inst(2))
make_pol(fd) = METHOD == "baseline1" ? make_policy_baseline1(fd) : make_policy_compact_lp(fd; H = 6)
pol1 = make_pol(fd1)
pol2 = make_pol(fd2)
θ2, _ = optimize_thresholds(fd2)

"Simulation of simulate_scenario (simulation.jl), with given draws U[t, τ, m]; counts the
periods where a component above its threshold was not maintained (dropped)."
function run(fd, policy, U, θ)
    x = ones(Int, fd.M); k = copy(fd.K0); total = 0.0; drops = 0; nmaint = 0
    for t in 1:(fd.inst.n_years * N_PERIODS_PER_YEAR)
        (t - 1) % N_PERIODS_PER_YEAR == 0 && (k = copy(fd.K0))
        col, _ = policy(x, k, t)
        check_action(fd, col, k)
        drops += count(m -> x[m] >= θ[m] && col.start[m] == 0, 1:fd.M)
        nmaint += count(>(0), col.start)
        pen, rep, x = simulate_period(fd, x, col, U[t, :, :])
        total += pen + rep
        k = k .- length.(col.free_days)
    end
    return total, drops, nmaint
end

@printf("method=%s  mtbf=%s  cc_scale=%.3f  scenarios=%d  thresholds=%s\n", METHOD, MTBF_MODE, CC_SCALE, N_SCEN, string(θ2[1:7]))
c2s = Float64[]; csum = Float64[]; d2s = 0; d1s = 0; m2s = 0
T_total = fd2.inst.n_years * N_PERIODS_PER_YEAR
for s in 1:N_SCEN
    U = rand(MersenneTwister(1000 + s), T_total, T_PERIOD, fd2.M)
    c2, d2, m2 = run(fd2, pol2, U, θ2)
    ca, da, _ = run(fd1, pol1, U[:, :, 1:7], θ2[1:7])
    cb, db, _ = run(fd1, pol1, U[:, :, 8:14], θ2[8:14])
    push!(c2s, c2); push!(csum, ca + cb)
    global d2s += d2; global d1s += da + db; global m2s += m2
    @printf("  scen %3d  two poles %9.3f  pole1+pole2 alone %9.3f  diff %+.3f\n", s, c2, ca + cb, c2 - ca - cb)
end
diff = c2s .- csum
@printf("\nmean cost two poles %.3f | mean sum of single poles %.3f | mean diff %+.3f (%.2f%%) | std diff %.3f\n",
        mean(c2s), mean(csum), mean(diff), 100 * mean(diff) / mean(csum), std(diff))
@printf("scenarios with |diff| > 1e-6: %d/%d | maintenances (two poles): %d | dropped (two poles): %d, (alone): %d\n",
        count(abs.(diff) .> 1e-6), N_SCEN, m2s, d2s, d1s)
