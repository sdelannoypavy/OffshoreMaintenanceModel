# tune_gamma.jl
#
# Tunes gamma (weight of the maintenance days in the per-component threshold optimisation of
# baseline 1, baseline1.jl) on a simplified instance. Usage:
#
#   julia --project=. paper_experiment/tune_gamma.jl [instance] [n_scenarios]
#
# gamma only acts through the thresholds θ (optimize_thresholds), which change at a finite
# number of gamma values: a fine grid of gamma is scanned, the distinct threshold vectors are
# kept, and each one is simulated on the SAME scenarios (common random numbers, as in
# run_experiment.jl). The best one gives the recommended gamma (the middle of the range of gamma
# that produces it, on the log scale for gamma > 0).

using Printf, Statistics

include(joinpath(@__DIR__, "instances.jl"))
include(joinpath(@__DIR__, "fluid_lp_cg.jl"))
include(joinpath(@__DIR__, "baseline1.jl"))
include(joinpath(@__DIR__, "simulation.jl"))

instance_name = length(ARGS) >= 1 ? ARGS[1] : "one_pole_simple"
n_scen = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200

fd = build_fluid_data(Dict(i.name => i for i in instances)[instance_name])
fd.inst.simplified || error("tune_gamma.jl is for the simplified instances")

gammas = vcat(0.0, exp.(range(log(0.01), log(1.0); length = 400)))
groups = Dict{Vector{Int},Vector{Float64}}()
for g in gammas
    θ, _ = optimize_thresholds(fd; gamma = g)
    push!(get!(groups, θ, Float64[]), g)
end
@printf("%d distinct threshold vectors for gamma in [0, 1]\n", length(groups))

results = NamedTuple[]
for (θ, gs) in sort(collect(groups); by = kv -> minimum(kv[2]))
    policy = make_policy_baseline1(fd; gamma = gs[1])
    costs = [simulate_scenario(fd, policy, s; method = "baseline1", instance = instance_name,
                               periods_file = nothing).total for s in 1:n_scen]
    push!(results, (θ = θ, gmin = minimum(gs), gmax = maximum(gs), mean = mean(costs),
                    se = std(costs) / sqrt(n_scen)))
    @printf("gamma in [%.3f, %.3f]  θ=%s  mean cost %.2f (se %.2f)\n", minimum(gs), maximum(gs),
            string(θ), results[end].mean, results[end].se)
end

best = results[argmin([r.mean for r in results])]
γ_best = best.gmin == 0.0 ? best.gmax / 2 : sqrt(best.gmin * best.gmax)
@printf("\nBest: θ=%s, gamma in [%.3f, %.3f], recommended gamma = %.3f, mean cost %.2f (se %.2f)\n",
        string(best.θ), best.gmin, best.gmax, γ_best, best.mean, best.se)
