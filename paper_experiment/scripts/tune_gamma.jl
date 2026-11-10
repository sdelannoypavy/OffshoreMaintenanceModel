# tune_gamma.jl
#
# Chooses the parameter γ of the two-step baseline 2SB (src/two_step_baseline.jl): the fraction of the
# cost of the maintenance days charged to a component alone in step 1. Usage (from the root of the
# repository):
#
#   julia --project=. paper_experiment/scripts/tune_gamma.jl <instance> [n_scenarios]
#
# γ only acts through the thresholds θ of step 1, which change at finitely many values of γ. A fine grid
# of γ ∈ [0, 1] is scanned, the distinct threshold vectors are kept, and each of them is simulated on the
# same scenarios 1..n_scenarios (200 by default). The best one gives the recommended γ, in the middle of
# the range of γ that produces it (on the log scale for γ > 0).

include(joinpath(@__DIR__, "..", "src", "setup.jl"))

instance = ARGS[1]
n_scen = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200
fd = build_fluid_data(INSTANCES[instance])

gammas = vcat(0.0, exp.(range(log(0.01), log(1.0); length = 400)))
groups = Dict{Vector{Int},Vector{Float64}}()   # threshold vector -> values of γ that produce it
for g in gammas
    θ, _ = optimize_thresholds(fd; gamma = g)
    push!(get!(groups, θ, Float64[]), g)
end
@printf("%d distinct threshold vectors for γ in [0, 1]\n", length(groups))

results = NamedTuple[]
for (θ, gs) in sort(collect(groups); by = kv -> minimum(kv[2]))
    policy = make_policy_two_step_baseline(fd; gamma = gs[1])
    costs = [simulate_scenario(fd, policy, s; method = "2SB", instance = instance, periods_file = nothing).total for s in 1:n_scen]
    push!(results, (θ = θ, gmin = minimum(gs), gmax = maximum(gs), mean = mean(costs), se = std(costs) / sqrt(n_scen)))
    @printf("γ in [%.3f, %.3f]  θ = %s  mean cost %.2f (se %.2f)\n", minimum(gs), maximum(gs), string(θ),
            results[end].mean, results[end].se)
end

best = results[argmin([r.mean for r in results])]
γ_best = best.gmin == 0.0 ? best.gmax / 2 : sqrt(best.gmin * best.gmax)
@printf("\nbest: θ = %s, γ in [%.3f, %.3f], recommended γ = %.3f, mean cost %.2f (se %.2f)\n",
        string(best.θ), best.gmin, best.gmax, γ_best, best.mean, best.se)
